from __future__ import annotations
import re
from typing import Optional
import pandas as pd

def setup_negation_pipeline():
    """
    Setup spaCy pipeline with negation detection.
    
    Returns:
        spacy.Language: Configured spaCy pipeline with negation detection
    """
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        print("[ERROR] Please install spaCy model: python -m spacy download en_core_web_sm")
        return None

    if NEGSPACY_AVAILABLE:
        try:
            # Remove existing negex pipe if it exists
            if "negex" in nlp.pipe_names:
                nlp.remove_pipe("negex")

            # Add negation detection with proper configuration
            nlp.add_pipe(
                "negex",
                config={
                    "ent_types": ["PRODUCT", "ORG", "DISEASE", "SYMPTOM"],
                    "chunk_prefix": ["no", "not", "without", "never", "denied", "lack of"],
                    "pseudo_negations": ["not only", "not just", "not necessarily"]
                }
            )
            print("[INFO] Negation pipeline configured with negspacy")
        except Exception as e:
            print(f"[WARNING] Failed to add negspacy: {e}")
            print("[INFO] Falling back to rule-based negation")
            return nlp
    else:
        print("[WARNING] negspacy not available, using basic negation rules")

    return nlp


def tag_negations_simple(text: str) -> str:
    """
    Simple rule-based negation tagging as fallback.
    
    Args:
        text (str): Text to tag negations in
        
    Returns:
        str: Text with [NEG] markers added to negation words
    """
    if pd.isna(text) or len(str(text)) == 0:
        return text

    text = str(text)

    # Simple negation patterns
    negation_patterns = [
        (r'\bno\b', '[NEG]no'),
        (r'\bnot\b', '[NEG]not'),
        (r'\bnever\b', '[NEG]never'),
        (r'\bwithout\b', '[NEG]without'),
        (r'\bdenied\b', '[NEG]denied'),
        (r"\bdidn't\b", "[NEG]didn't"),
        (r"\bdon't\b", "[NEG]don't"),
        (r"\bwon't\b", "[NEG]won't"),
        (r"\bcan't\b", "[NEG]can't"),
        (r"\bcannot\b", "[NEG]cannot"),
        (r"\bhadn't\b", "[NEG]hadn't"),
        (r"\bhasn't\b", "[NEG]hasn't"),
        (r"\bhaven't\b", "[NEG]haven't"),
        (r"\bisn't\b", "[NEG]isn't"),
        (r"\bwasn't\b", "[NEG]wasn't"),
        (r"\bweren't\b", "[NEG]weren't"),
        (r"\bwouldn't\b", "[NEG]wouldn't"),
        (r'\bnone\b', '[NEG]none'),
        (r'\bneither\b', '[NEG]neither'),
        (r'\bnor\b', '[NEG]nor'),
    ]

    for pattern, replacement in negation_patterns:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    return text


def tag_negations(text: str, nlp) -> str:
    """
    Tag negations in text using spaCy + negspacy.
    
    Args:
        text (str): Text to tag negations in
        nlp: spaCy language model with negation detection
        
    Returns:
        str: Text with [NEG] markers added to negated tokens
    """
    if pd.isna(text) or len(str(text)) == 0:
        return text

    try:
        doc = nlp(str(text))

        # Check if negex extension is available
        if hasattr(doc[0]._, 'negex'):
            # Build negated text with markers
            tokens_with_neg = []
            for token in doc:
                if token._.negex:  # If token is negated
                    tokens_with_neg.append(f"[NEG]{token.text}")
                else:
                    tokens_with_neg.append(token.text)

            return ' '.join(tokens_with_neg)
        else:
            # Fallback to simple rule-based
            return tag_negations_simple(text)
    except Exception:
        # If any error, use simple rule-based
        return tag_negations_simple(text)


def apply_negation_tagging(dataframe: pd.DataFrame, text_column: str = 'text_clean') -> pd.DataFrame:
    """
    Apply negation tagging to all reviews in the dataframe.
    
    Args:
        dataframe (pd.DataFrame): Input dataframe with cleaned text
        text_column (str): Name of the column containing cleaned text
        
    Returns:
        pd.DataFrame: Dataframe with negation-tagged text in 'text_neg' column
    """
    print(f"[INFO] Applying negation tagging to '{text_column}' column...")

    # Try to setup spaCy pipeline
    nlp = setup_negation_pipeline()

    # Determine which method to use
    use_spacy = nlp is not None and NEGSPACY_AVAILABLE

    if use_spacy:
        print(f"[INFO] Using spaCy + negspacy for negation detection")
        # Test on first text to see if it works
        try:
            test_text = dataframe[text_column].iloc[0]
            _ = tag_negations(test_text, nlp)
            print(f"[INFO] Negation pipeline test successful")
        except Exception as e:
            print(f"[WARNING] Negation pipeline test failed: {e}")
            print(f"[INFO] Falling back to simple rule-based negation")
            use_spacy = False
    else:
        print(f"[INFO] Using simple rule-based negation detection")
        use_spacy = False

    # Apply negation tagging
    negated_texts = []
    total = len(dataframe)

    for idx, text in enumerate(dataframe[text_column]):
        if idx % 5000 == 0:
            print(f"[INFO] Tagging negations: {idx}/{total} records... ({idx/total*100:.1f}%)")

        if use_spacy:
            try:
                neg_text = tag_negations(text, nlp)
            except:
                neg_text = tag_negations_simple(text)
        else:
            neg_text = tag_negations_simple(text)

        negated_texts.append(neg_text)

    dataframe['text_neg'] = negated_texts
    print(f"[INFO] Negation tagging completed")

    # Show some examples
    print("\n[INFO] Sample negation tagging results:")
    for i in range(min(3, len(dataframe))):
        if '[NEG]' in negated_texts[i]:
            print(f"\nOriginal: {dataframe[text_column].iloc[i][:200]}...")
            print(f"Tagged: {negated_texts[i][:200]}...")
            break

    return dataframe
