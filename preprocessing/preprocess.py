"""
Full preprocessing pipeline for titles, abstracts and fulltexts.
Creates a "light" version of the OpenAlex and CORE objects:
- only keeps essential metadata
- cleans texts from encoding errors and HTML/XML and LaTeX artifacts
- discards non-English texts
- discards objects that do not have either abstract or fulltext
"""

from dataclasses import dataclass, asdict
from typing import Literal
import gzip
import json
from pathlib import Path
import logging
import unicodedata
import regex
import re
from lingua import Language, LanguageDetectorBuilder
import ftfy
from functools import reduce

# ---------------------------------------------------------------------------
# Logger setup
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s (line %(lineno)d) | %(levelname)s: %(message)s"
    )

@dataclass
class PreprocessingMetadata:
    index: int
    source_file: str
    id: str
    status: Literal["skipped", "error"]
    skip_reason: str | None = None
    error_type: str | None = None
    error_msg: str | None = None

class PreprocessingError(Exception):
    """Base class"""
    reason: str

class NotEnglishError(PreprocessingError):
    reason = "not_english"

class NoContentError(PreprocessingError):
    def __init__(self, field: str):
        self.field = field
        self.reason = f"empty_field:{field}"
        super().__init__(f"Field '{field}' is empty")

class NoMatchingDoiError(PreprocessingError):
    reason = "doi_not_in_oa"

# ---------------------------------------------------------------------------
# Language detection
# ---------------------------------------------------------------------------

languages = [Language.ENGLISH]
detector = LanguageDetectorBuilder.from_languages(*languages).build()

def is_english(text: str) -> bool | None:
    """Only checks if the first 1000 characters of a text are English. If not, returns False.
    If text is not passed for the function, returns None to signal that detector has not been used."""
    if not text:
        return None
    language = detector.detect_language_of(text[:1000]) # only use max 1000 chars for detection
    if not language:
        return False
    return True


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

# Map for replacing certain characters
REPLACE_MAP = {
    '\u00ad': '-',       # soft hyphen to hyphen
    '\uf02a': '*',       # asterisk
    '\uf02d': '-',       # hyphen
    '\uf0b7': '\u2022',  # bullet •
    '\uf0a7': '\u2022',  # bullet variant •
}

TO_REPLACE_PATTERN = regex.compile(r'\p{Zs}|\p{Cf}|\p{Co}') # to find all characters in these Unicode categories
TAG_PATTERN = regex.compile(r'<[\/!a-z]+\w*?[^()<>,]*?>') # find HTML and XML like tags and remove them
MULTIPLE_SPACE_PATTER = regex.compile(r' {2,}') # to find space sequences and replace them with a single space
MULTIPLE_LINEBREAK_PATTERN = regex.compile(r'(?:\n\s*){2,}') # catch occasions where line break appears with some other whitespace character
SINGLE_LINEBREAK_PATTERN = regex.compile(r'(?<!\n)\n(?!\n)') # only match line breaks that appear with no other line breaks
    
def replace_char(match) -> str:
    """Normalizes or removes the characters that ftfy alone doesn't:
    - normalize caharcters from Unicode category Zs to a space
    - replaces or removes characters from categories Cf and Co according to a mapping"""
    char = match.group(0)
    cat = unicodedata.category(char)

    # Normalize spaces
    if cat == "Zs": # Space_Separator
        return " "
    # Replace from mapping or drop
    if cat in ("Cf", "Co"): # Format, Private_Use
        return REPLACE_MAP.get(char, "")
    return ""

def replace_chars(text: str) -> str:
    return TO_REPLACE_PATTERN.sub(replace_char, text)

def remove_html_tags(text: str) -> str:
    return TAG_PATTERN.sub("", text)

def unescape_linebreaks(text: str) -> str:
    return text.replace("\\n", "\n")

def fix_linebreaks(text: str) -> str:
    text = text.replace("-\n", "")
    return SINGLE_LINEBREAK_PATTERN.sub(" ", text)

def normalize_text(text: str) -> str:
    return unicodedata.normalize("NFKC", text)

def normalize_space(text: str) -> str:
    text = MULTIPLE_SPACE_PATTER.sub(" ", text)
    text = MULTIPLE_LINEBREAK_PATTERN.sub("\n\n", text)
    return text.strip()

def normalize_abstract_space(text: str):
    return " ".join(text.split()).strip()


# ---------------------------------------------------------------------------
# LaTeX cleaning
# ---------------------------------------------------------------------------

# For replacing certain LaTeX commands with their unicode variants
SYMBOL_MAP = {
    # LaTeX special characters
    r"\%":   "%",
    r"\$":   "$",
    r"\&":   "&",
    r"\#":   "#",
    r"\{":   "{",
    r"\}":   "}",

    # Common commands
    r"\times":          "×",
    r"\rightarrow":     "→",
    r"\leftarrow":      "←",
    r"\Rightarrow":     "⇒",
    r"\Leftarrow":      "⇐",
    r"\leftrightarrow": "↔",
    r"\sim":            "∼",
    r"\approx":         "≈",
    r"\leq":            "≤",
    r"\geq":            "≥",
    r"\neq":            "≠",
    r"\pm":             "±",
    r"\infty":          "∞",
    r"\in":             "∈",
    r"\notin":          "∉",
    r"\ell":            "ℓ",
    r"\log":            "log",
    r"\cdot":           "·",
    r"\ldots":          "…",
    r"\dots":           "…",
    r"\circ":           "°",
    r"\eg":             "e.g.",
    r"\ie":             "i.e.",
    r"\vs":             "vs.",
    r"\textquotesingle ":"'",
    r"\langle":         "⟨",
    r"\rangle":         "⟩",
    r"\lfloor":         "⌊",  # U+230A
    r"\rfloor":         "⌋",  # U+230B
    r"\lceil":          "⌈",  # U+2308
    r"\rceil":          "⌉",  # U+2309
    r"\lvert":          "|",
    r"\rvert":          "|",
    r"\lVert":          "‖",  # U+2016 double vertical line
    r"\rVert":          "‖",

    # Lowercase Greek
    r"\alpha":   "α",
    r"\beta":    "β",
    r"\gamma":   "γ",
    r"\delta":   "δ",
    r"\epsilon": "ε",
    r"\varepsilon": "ε",
    r"\zeta":    "ζ",
    r"\eta":     "η",
    r"\theta":   "θ",
    r"\vartheta":"θ",
    r"\iota":    "ι",
    r"\kappa":   "κ",
    r"\lambda":  "λ",
    r"\mu":      "μ",
    r"\nu":      "ν",
    r"\xi":      "ξ",
    r"\pi":      "π",
    r"\varpi":   "π",
    r"\rho":     "ρ",
    r"\varrho":  "ρ",
    r"\sigma":   "σ",
    r"\varsigma":"ς",
    r"\tau":     "τ",
    r"\upsilon": "υ",
    r"\phi":     "φ",
    r"\varphi":  "φ",
    r"\chi":     "χ",
    r"\psi":     "ψ",
    r"\omega":   "ω",

    # Uppercase Greek (only those that differ from Latin letters)
    r"\Gamma":   "Γ",
    r"\Delta":   "Δ",
    r"\Theta":   "Θ",
    r"\Lambda":  "Λ",
    r"\Xi":      "Ξ",
    r"\Pi":      "Π",
    r"\Sigma":   "Σ",
    r"\Upsilon": "Υ",
    r"\Phi":     "Φ",
    r"\Psi":     "Ψ",
    r"\Omega":   "Ω",
}

# Frequently appearing commands
PASSTHROUGH_CMDS = {
    "textbf", "textit", "texttt", "textsc", "textrm", "textsf"
    "emph", "underline",
    "mathbf", "mathcal", "mathbb", "mathrm", "mathsf", "mathscr", "mathtt"
    "boldsymbol",
    "text",
    "url",
    "it", "em", "rm", "tt",
    "footnote",
    "cite", "citep", "citet",
}

def _replace_unicode_cmd(m: re.Match) -> str:
    code = m.group(1).strip()
    try:
        if code.lower().startswith("x"):
            return chr(int(code[1:], 16))
        return chr(int(code))
    except (ValueError, OverflowError):
        return m.group(0)

def _clean_latex_inner(text: str) -> str:

    # Normalize double backslashes to just one - data is unlikely to include e.g.
    # file paths that would break because of this
    text = text.replace("\\\\", "\\")

    # Remove math mode delimiters - unlikey for these sequences to appear for other
    # reasons
    text = re.sub(r"\\\(", "", text)
    text = re.sub(r"\\\)", "", text)
    text = re.sub(r"\\\[", "", text)
    text = re.sub(r"\\\]", "", text)

    # Replace unicode commands with the actual characters
    text = re.sub(r"\\unicode\{([^}]+)\}", _replace_unicode_cmd, text)

    # Replace fraction commands, e.g. \frac{1}{2} → 1⁄2
    text = re.sub(r"\\frac\{([^{}]*)\}\{([^{}]*)\}", r"\1⁄\2", text)

    # Extract link descriptions, provide url in parentheses
    text = re.sub(r"\\href\{([^{}]*)\}\{([^{}]*)\}", r"\2 (\1)", text)

    # Replace sqrt (square root) command 
    text = re.sub(r"\\sqrt\{([^{}]*)\}", r"√\1", text)

    # Remove textcolor commands, keep content
    text = re.sub(r"\\textcolor\{([^{}]*)}\{([^{}]*)\}", r"\2", text)

    for cmd in PASSTHROUGH_CMDS:
        text = re.sub(rf"\\{re.escape(cmd)}\{{([^{{}}]*)\}}", r"\1", text)
        text = re.sub(rf"\{{\\{re.escape(cmd)}\s+([^{{}}]*?)\}}", r"\1", text)

    for latex, replacement in sorted(SYMBOL_MAP.items(), key=lambda x: -len(x[0])):
        text = text.replace(latex, replacement)

    # Unknown \cmd{content} → content
    text = re.sub(r"\\[A-Za-z]+\{([^{}]*)\}", r"\1", text)

    # Bare unknown commands: only before whitespace/punctuation
    # Require at least one letter to avoid eating lone backslashes
    text = re.sub(r"\\[A-Za-z]+(?=[\s,.\);:\-]|$)", "", text)

    return text

# Regex for identifying when dollar signs signal LaTeX math mode and removing them -
# will not catch everything, but shouldn't produce too many false positives
_MATH_DOLLAR_RE = re.compile(
    r"""
    # Double dollars signal math mode pretty reliably
    (?<![A-Za-z0-9\\])  # not preceded by alphanumeric or backslash
    \$\$                # opening $$
    (.*?)               # content (group 1)
    \$\$                # closing $$
    |
    # If single dollars, only match content that does not include whitespace without extra rules
    \$
    ([^\s$]+)           # content (group 2), not allowed to include whitespace
    \$
    |
    # If single dollars, be more strict about rules if content includes whitespace
    (?<![A-Za-z0-9\\])  # not preceded by alphanumeric or backslash
    \$                  # opening $
    ([^$]{1,100}?)      # content (group 3), allowed to include whitespace
    \$
    (?![0-9])           # closing $ not followed by digit
    """,
    re.VERBOSE | re.DOTALL,
)

def _replace_any_math(m: re.Match) -> str:
    # Find the matched group
    inner = next(g for g in m.groups() if g is not None)
    return inner

def clean_latex(text: str, inspect_mode: bool = False) -> str:
    text = _MATH_DOLLAR_RE.sub(
        lambda m: _replace_any_math(m),
        text,
    )
    cleaned_text = _clean_latex_inner(text)
    return cleaned_text


# ---------------------------------------------------------------------------
# Pipeline build
# ---------------------------------------------------------------------------

def pipe(*functions):
    """Apply all functions given as a parameter from left to right."""
    def pipeline(text):
        return reduce(lambda t, f: f(t), functions, text)
    return pipeline

fix_abstract = pipe(
    ftfy.fix_text,
    replace_chars,
    remove_html_tags,
    unescape_linebreaks,
    fix_linebreaks,
    clean_latex,
    normalize_text,
    normalize_abstract_space,
)

fix_fulltext = pipe(
    ftfy.fix_text,
    replace_chars,
    remove_html_tags,
    unescape_linebreaks,
    fix_linebreaks,
    clean_latex,
    normalize_text,
    normalize_space,
)

# ---------------------------------------------------------------------------
# Data extraction
# ---------------------------------------------------------------------------

def validate_inverted_index(inverted_index: dict[str, list[int]] | None) -> bool:

    if not isinstance(inverted_index, dict):
        return False
    
    for key, value in inverted_index.items():
        if not isinstance(key, str):
            return False
        if not isinstance(value, list):
            return False
        if not all(isinstance(i, int) for i in value):
            return False
    
    return True

def inverse_index(index: dict[str, list[int]]) -> str:
    abstract_dict = {}
    for token, index_list in index.items():
        for i in index_list:
            abstract_dict[i] = token

    tokens = []
    for i in range(max(abstract_dict)+1):
        token = abstract_dict.get(i)
        if token:
            tokens.append(token)
    abstract = " ".join(tokens)
        
    return abstract

def get_topic(topic: dict | None) -> str | None:
    if not topic:
        return None
    return topic.get("display_name")

def get_inner_value(topic: dict | None, outer_key: str) -> str | None:
    if not topic:
        return None
    if isinstance(topic, dict) and topic.get(outer_key):
        return topic[outer_key].get("display_name")

# ---------------------------------------------------------------------------
# Object construction
# ---------------------------------------------------------------------------

def create_oa_light_version(obj: dict) -> dict | None:

    abstract_inverted_index = obj.get("abstract_inverted_index", {})
    abstract = inverse_index(abstract_inverted_index) if validate_inverted_index(abstract_inverted_index) else None

    if abstract == None:
        raise NoContentError("abstract")

    abstract = fix_abstract(abstract)

    english = is_english(abstract) if abstract else None
    if not english:
        raise NotEnglishError()

    title = obj.get("title")
    if title:
        title = fix_abstract(title)

    primary_topic = obj.get("primary_topic")
    has_content = obj.get("has_content", {})
    if not isinstance(has_content, dict):
        has_content = {}
    best_oa_location = obj.get("best_oa_location", {})
    if not isinstance(best_oa_location, dict):
        best_oa_location = {}

    id = obj.get("id")
    if id:
        id = id.replace("https://openalex.org/", "")

    new_obj = {
        "id": id,
        "doi": obj.get("doi"),
        "title": title,
        "publication_year": obj.get("publication_year"),
        "publication_date": obj.get("publication_date"),
        "updated_date": obj.get("updated_date"),
        "created_date": obj.get("created_date"),
        "document_type": obj.get("type"),
        "topic": get_topic(primary_topic),
        "subfield": get_inner_value(primary_topic, "subfield"),
        "field": get_inner_value(primary_topic, "field"),
        "domain": get_inner_value(primary_topic, "domain"),
        "abstract": abstract,
        "has_xml": has_content.get("grobid_xml", False),
        "pdf_url": best_oa_location.get("pdf_url"),
        "license": best_oa_location.get("license"),
        "source": "oa"
    }
    return new_obj

def create_core_light_version(obj: dict) -> dict | None:

    abstract = obj.get("abstract")
    fulltext = obj.get("fullText")
    title = obj.get("title", "")

    if fulltext == None:
        raise NoContentError("fulltext")
    fulltext = fix_fulltext(fulltext)

    english = is_english(fulltext) if fulltext else None
    if not english:
        raise NotEnglishError()

    if abstract:
        abstract = fix_abstract(abstract)

    if title:
        title = fix_abstract(title)
    
    new_obj = {
        "id": obj.get("id"),
        "doi": obj.get("doi"),
        "title": title,
        "publication_year": obj.get("yearPublished"),
        "publication_date": obj.get("publishedDate"),
        "updated_date": obj.get("updatedDate"),
        "created_date": obj.get("createdDate"),
        "document_type": obj.get("documentType"),
        "field": obj.get("fieldOfStudy"),
        "abstract": abstract,
        "fulltext": fulltext,
        "pdf_urls": obj.get("sourceFulltextUrls"),
        "source": "core"
    }
    return new_obj

def yield_objects(filename: str | Path):
    with gzip.open(filename) as file:
        for line in file:
            obj = json.loads(line)
            yield obj

# ---------------------------------------------------------------------------
# Match CORE with OpenAlex
# ---------------------------------------------------------------------------

def get_doi(obj: dict) -> str:
    doi = obj.get("doi", "") or ""
    doi = str(doi).replace("https://doi.org/", "").strip()
    return doi

def get_oa_dois():
    OA_DEDUPE_FILEPATH = "/scratch/project_xxxxxxx/data_collection/data/light-data/light-new-query-dedupe.jsonl.gz"
    
    oa_dois = set()
    for oa_obj in yield_objects(OA_DEDUPE_FILEPATH):
        doi = get_doi(oa_obj)
        if doi:
            oa_dois.add(doi)
    return oa_dois

# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():

    # Local tests
    # FILENAME = "data/openalex-articles_000.jsonl.gz"
    # FILENAME = "data/core-fulltexts-test.jsonl.gz"

    SOURCE_DIR = "/scratch/project_xxxxxxx/data_collection/data/jsonl-data"
    DEST_FILE = "/scratch/project_xxxxxxx/data_collection/data/light-data/light-new-query-core.jsonl.gz"
    ERROR_LOG_FILE = "/scratch/project_xxxxxxx/data_collection/data/inspection-data/preprocessing-log-new-query-core.jsonl"

    OA_DOIS = get_oa_dois()

    errors = 0
    skipped = 0
    total = 0

    with gzip.open(DEST_FILE, "wt", encoding="utf-8") as file, open(ERROR_LOG_FILE, "a") as error_log:
        for file_path in sorted(Path(SOURCE_DIR).glob("core*.jsonl.gz")):

            is_oa = True if "openalex" in file_path.name or "oa" in file_path.name else False

            logger.info(f"Reading file {file_path.name}")
            preprocessing_fn = create_oa_light_version if is_oa else create_core_light_version
            interval = 5000 if is_oa else 500

            for i, obj in enumerate(yield_objects(file_path), start=1):

                total += 1
                if i % interval == 0:
                    logger.info(f"{i} documents processed")
                    error_log.flush()
                
                try:
                    core_doi = get_doi(obj)
                    if core_doi not in OA_DOIS:
                        raise NoMatchingDoiError()
                    new_obj = preprocessing_fn(obj)
                    file.write(json.dumps(new_obj, ensure_ascii=False))
                    file.write("\n")
                except PreprocessingError as e:
                    skipped += 1
                    obj_id = obj.get("id", "<NO_ID>")
                    if isinstance(obj_id, str):
                        obj_id = obj_id.replace("https://openalex.org/", "")
                    meta = PreprocessingMetadata(i, file_path.name, obj_id, "skipped", skip_reason=e.reason)
                    error_log.write(json.dumps(asdict(meta))+ "\n")
                except Exception as e:
                    errors += 1
                    obj_id = obj.get("id", "<NO_ID>")
                    if isinstance(obj_id, str):
                        obj_id = obj_id.replace("https://openalex.org/", "")
                    meta = PreprocessingMetadata(i, file_path.name, obj_id, "error",
                                                error_type=type(e).__name__, error_msg=str(e))
                    error_log.write(json.dumps(asdict(meta))+ "\n")
    
    logger.info(f"{total-errors-skipped}/{total} ({(total-errors-skipped)/total:.2%}) succeeded")
    logger.info(f"{skipped}/{total} ({skipped/total:.2%}) skipped")
    logger.info(f"{errors}/{total} ({errors/total:.2%}) failed")


if __name__ == "__main__":
    main()