import logging
import os
import argparse
import anthropic
from pathlib import Path
from dotenv import load_dotenv

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

load_dotenv()

class MarkdownTranslator:
    # Language code mapping
    LANG_CODES = {
        'English': 'en',
        'Japanese': 'ja',
        'Chinese': 'zh'
    }
    
    def __init__(self, api_key):
        self.client = anthropic.Client(api_key=api_key)
        
    def read_markdown_file(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
            
    def write_markdown_file(self, content, output_path):
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
            
    def translate_content(self, content, source_lang, target_langs):
        # Construct the prompt
        logger.info(f"Translating content from {source_lang} to {', '.join(target_langs)}")
        target_lang = ', '.join(target_langs)
        prompt = f"""Here is a markdown file in {source_lang}:

      {content}

      Please translate this markdown file into the following languages while preserving all frontmatter, formatting, and structure: {', '.join(target_langs)}

      For each translation, please maintain:
      1. All frontmatter fields and values (only translate the title if appropriate)
      2. All markdown formatting (headers, lists, bold text, etc.)
      3. Any links or image references
      4. The overall document structure

      Please add a clear delimiter "# Translated Version: {target_lang}" at the top line of each translation."""

        # Make the API call
        response = self.client.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=4096,
            temperature=0,
            system="You are a helpful assistant that specializes in translating content while preserving markdown formatting.",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        logger.info(f"API response: OK")
        
        return response.content[0].text.strip()
    
    def split_translations(self, translations_text, target_langs):
        """Split the combined translations into separate files."""
        translations = {}
        current_lang = None
        current_content = []
        
        logger.info(f"Splitting translations with length of {len(translations_text)} into separate files")
        for line in translations_text.split('\n'):
            # Check for language version headers
            for lang in target_langs:
                if f"# Translated Version: {lang}" in line:
                    if current_lang and current_content:
                        translations[current_lang] = '\n'.join(current_content).strip()
                    current_lang = lang
                    current_content = []
                    break
            else:
                if current_lang:
                    current_content.append(line)
                    
        # Add the last translation
        if current_lang and current_content:
            translations[current_lang] = '\n'.join(current_content).strip()
        
        logger.info(f"Translated into {translations.keys() if translations else 'No translations found'}")
        
        return translations

def main():
    parser = argparse.ArgumentParser(description='Translate markdown files using Claude API')
    parser.add_argument('input_file', help='Path to input markdown file')
    parser.add_argument('--source', help='Source language', default='Chinese')
    parser.add_argument('--target', nargs='+', help='Target languages', default=['English', 'Japanese'])
    parser.add_argument('--output_dir', help='Output directory', default='../content/posts')
    args = parser.parse_args()

    # Get API key from environment variable
    api_key = os.getenv('ANTHROPIC_API_KEY')
    if not api_key:
        raise ValueError("Please set ANTHROPIC_API_KEY environment variable")

    # Initialize translator
    translator = MarkdownTranslator(api_key)

    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Read input file
    content = translator.read_markdown_file(args.input_file)

    try:
        # Get translations
        translations_text = translator.translate_content(content, args.source, args.target)
        
        # Split translations into separate files
        translations = translator.split_translations(translations_text, args.target)

        logger.info(f"Translations: {translations.get('English', 'No translations found'), translations.get('Japanese', 'No translations found')}")
        # Write translations to files
        input_filename = Path(args.input_file).stem.split('.')[0]  # Remove any existing language suffix
        logger.info(f"Writing translations to files...{input_filename}")
        for lang, trans_content in translations.items():
            lang_code = translator.LANG_CODES.get(lang, lang.lower())
            output_file = output_dir / f"{input_filename}.{lang_code}.md"
            logger.info(f"Writing translation to file: {output_file}")
            translator.write_markdown_file(trans_content, output_file)
            logger.info(f"Translated file saved to: {output_file}")

    except Exception as e:
        logger.error(f"Error during translation: {str(e)}")

if __name__ == "__main__":
    main()