import json
import regex as re
import os
import logging
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import copy
import argparse

def remove_redundant_braces(text):
    pattern = r"{{(?:[^{}]*|(?R))*}}"
    matches = list(re.finditer(pattern, text))

    delete_ranges = []
    for match in matches:
        start = match.start()
        end = match.end()
        if start == 0 or (start > 0 and text[start - 1] == "\n") or end == len(text) or (end < len(text) and text[end] == "\n"):
            delete_ranges.append((match.start(), match.end()))

    new_text = []
    prev_end = 0

    for start, end in delete_ranges:
        new_text.append(text[prev_end:start])
        prev_end = end

    new_text.append(text[prev_end:])
    final_text = ''.join(new_text)
    return final_text

def clean_text(text):
    # remove redundant braces
    # ===Useful: mostly appear with text paragraph===
    # {{Coord|-11.05| 15.0822|name=Bridge 14}}
    # {{convert|6|ft|3|in|cm|abbr=on}} tall and weighs {{convert|220|lb|kg|abbr=on}}
    # in the Mexican League in {{Baseball year|2007}}
    # Yedioth Tel Aviv ({{lang-he|ידיעות תל אביב}})
    # Lothian books {{ISBN|0-7344-0590-1}}
    # ===Useless: mostly appear as single lines===
    # {{1970s-comedy-film-stub}}
    # {{Poland-film-stub}}
    # {{Infobox...}}
    text = remove_redundant_braces(text)

    # remove {{cite web|url=...}} --> [16][17] with links and text
    # {{citation|title=...}} with only text
    text = re.sub(r"\{\{[cC]ite ?.*?\}\}", '', text, flags=re.DOTALL)
    text = re.sub(r"\{\{[cC]itation.*?\}\}", '', text, flags=re.DOTALL)

    # remove tables double or single layers
    text = re.sub(r"\{\|(?:[^{}]|\{[^{}]*\})*\|\}", "", text, flags=re.DOTALL)
    text = re.sub(r"\{\|.*?\|\}", "", text, flags=re.DOTALL)

    # remove categories links at the bottom
    text = re.sub(r"\[\[Category:.*?\]\]", "", text)

    # remove gallery tags
    text = re.sub(r"<gallery>.*?</gallery>", "", text, flags=re.DOTALL)
    text = re.sub(r"<gallery[^>/]*?>.*?</gallery>", "", text, flags=re.DOTALL)
    # remove files and images
    text = re.sub(r"^\s*\[\[[Ff]ile:.*?\]\]\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"\[\[[Ff]ile:.*?\]\]", "", text, flags=re.DOTALL)
    text = re.sub(r"^\s*\[\[[Ii]mage:.*?\]\]\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"\[\[[Ii]mage:.*?\]\]", "", text, flags=re.DOTALL)

    # remove ref tags
    # &lt;ref&gt;...&lt;/ref&gt;--><ref>...</ref>
    # e.g. &lt;ref&gt;{{cite web |...}}
    text = re.sub(r"<ref>.*?</ref>", "", text, flags=re.DOTALL)
    text = re.sub(r"<ref[^>/]*?>.*?</ref>", "", text, flags=re.DOTALL)
    text = re.sub(r"&lt;ref&gt;.*?&lt;/ref&gt;", "", text, flags=re.DOTALL)

    # remove paired html tags
    text = re.sub(r"<.*?/>", "", text)
    # remove single html tags    &lt;.*?&gt;--><...>    <!--.*?-->
    text = re.sub(r"<.*?>", "", text, flags=re.DOTALL)
    text = re.sub(r"&lt;.*?&gt;", "", text, flags=re.DOTALL)

    # remove html entities
    # e.g. &quot;  &amp;  &nbsp;  &lt;
    text = re.sub(r"&[a-zA-Z0-9#]+;", "", text)

    # remove html links and save last words if exist
    # [https://www.wsj.com/articles... Wall Street Journal]
    # [https://www.wsj.com/articles...]
    text = re.sub(r"\[https?:\/\/[^\] ]+(?: ([^\]]+))?\]", lambda m: m.group(1) if m.group(1) else "", text)

    # remove single line start with | ! { }
    text = re.sub(r"^\s*[|!{}].*$", "", text, flags=re.MULTILINE)

    # remove extra \n
    text = re.sub(r"\n+", "\n", text).strip()

    return text

def extract_text_links(text):
    see_also_docs = []
    external_links_docs = []

    parts = text.split('\n')
    length = len(parts)
    index = 0

    while index < length:
        part = parts[index].strip()
        if not part:
            index += 1
            continue

        match_title = re.match(r'(={2,})\s*(.*?)\s*\1', part)

        if not match_title:
            index += 1
            continue

        # the part is a title
        level = len(match_title.group(1))
        raw_text = match_title.group(2).strip()

        if "external links" in raw_text.lower():
            index += 1
            while index < length:
                part = parts[index].strip()
                match_title = re.match(r'(={2,})\s*(.*?)\s*\1', part)
                if match_title:
                    break
                
                if not part:
                    index += 1
                    continue

                if part.startswith('{{') and part.endswith('}}'):
                    inner_text = part[2:-2].strip()
                    excluded_keywords = ['Authority control', 'DEFAULTSORT:', 'commonscat-inline', 's-start', 's-off', 's-end', '|', '{', '}', 'Commons category']
                    
                    # {{Rheinmetall|state=collapsed}}
                    if '|state=' in inner_text:
                        wiki_title = inner_text.split('|')[0].strip()
                        external_links_docs.append(wiki_title)
                    else:
                        if not any(keyword.lower() in inner_text.lower() for keyword in excluded_keywords):
                            external_links_docs.append(inner_text)
                        
                            
                index += 1
                
        elif "see also" in raw_text.lower():
            index += 1
            while index < length:
                part = parts[index].strip()
                match_title = re.match(r'(={2,})\s*(.*?)\s*\1', part)
                if match_title:
                    break
                
                if not part:
                    index += 1
                    continue
                
                excluded_keywords = ['[', ']', '{', '}']

                if part.startswith('*'):
                    wiki_links = re.findall(r'\[\[([^\]]+)\]\]', part)
            
                    for link in wiki_links:
                        if any(keyword in link for keyword in excluded_keywords):
                            index += 1
                            continue
                        if '|' in link:
                            wiki_title = link.split('|')[-1].strip()
                        else:
                            wiki_title = link.strip()
                        
                        if wiki_title:
                            see_also_docs.append(wiki_title)
                
                index += 1
        else:
            index += 1

    return see_also_docs, external_links_docs


def add_intro_links(text):
    linked_docs = []

    cleaned_text = clean_text(text)

    parts = cleaned_text.split('\n')
    length = len(parts)
    index = 0

    while index < length:
        part = parts[index].strip()
        if not part:
            index += 1
            continue

        match_title = re.match(r'(={2,})\s*(.*?)\s*\1', part)

        if match_title:
            break

        wiki_links = re.findall(r'\[\[([^|\]]+\|)?([^\]]+)\]\]', part)
        
        # [[xxx]] [[xxx|xxx]]
        for link in wiki_links:
            wiki_title = link[1].strip()
            if wiki_title:
                linked_docs.append(wiki_title)

        index += 1

    return linked_docs


def extract_linked_docs(jsonl_file, output_jsonl_file, title_lower_to_original):
    with open(jsonl_file, 'r', encoding='utf-8') as f, open(output_jsonl_file, 'w', encoding='utf-8') as out_f:

        for line in f:
            data = json.loads(line)
            raw_text = data.get('raw_text')
            raw_see_also_docs, raw_external_links_docs = extract_text_links(raw_text)
            raw_intro_docs = add_intro_links(raw_text)

            see_also_docs = []
            for doc in raw_see_also_docs:
                original_title = title_lower_to_original.get(doc.lower())
                if original_title and original_title not in see_also_docs:
                    see_also_docs.append(original_title) 

            external_links_docs = []
            for doc in raw_external_links_docs:
                original_title = title_lower_to_original.get(doc.lower())
                if original_title and original_title not in external_links_docs:
                    external_links_docs.append(original_title) 

            intro_docs = []
            for doc in raw_intro_docs:
                original_title = title_lower_to_original.get(doc.lower())
                if original_title and original_title not in intro_docs:
                    intro_docs.append(original_title) 

            output_data = {
                "title": data.get('title'),
                "num_nodes": data.get('num_nodes'),
                "extracted_nodes": data.get('extracted_nodes'),
                "full_text": data.get('full_text'),
                "see_also_docs": see_also_docs,
                "external_links_docs": external_links_docs,
                "intro_docs": intro_docs,
                "raw_see_also_docs": raw_see_also_docs,
                "raw_external_links_docs": raw_external_links_docs,
                "raw_intro_docs": raw_intro_docs
            }

            out_f.write(json.dumps(output_data, ensure_ascii=False) + '\n')



def process_single_file(jsonl_file, output_folder, title_lower_to_original):
    try:
        filename = os.path.basename(jsonl_file)
        output_jsonl_file = os.path.join(output_folder, filename)
        extract_linked_docs(jsonl_file, output_jsonl_file, title_lower_to_original)
        logging.info(f"===Successfully process {output_jsonl_file} ===")

    except Exception as e:
        logging.error(f"Error processing {jsonl_file}: {e}")
        logging.error(f"Error details: {str(e)}", exc_info=True)

def process_files(input_folder, output_folder, index_file, max_workers=None):
    with open(index_file, 'r', encoding='utf-8') as f:
        title_index = json.load(f)
        title_lower_to_original = {title.lower(): title for title in title_index.keys()}

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    jsonl_files = [os.path.join(input_folder, filename) for filename in os.listdir(input_folder) if '.jsonl' in filename]

    if max_workers is None:
        max_workers = cpu_count()
    logging.info(f"Using {max_workers} workers for processing.")

    with Pool(processes=max_workers) as pool:
        list(tqdm(
            pool.starmap(process_single_file, [(jsonl_file, output_folder, title_lower_to_original) for jsonl_file in jsonl_files]),
            total=len(jsonl_files),
            desc="Processing files",
            unit="file"
        ))

    logging.info("All files processed successfully.")

def main():
    parser = argparse.ArgumentParser(description="Attach Wikipedia link metadata to parsed document trees.")
    parser.add_argument("--input_folder", required=True, help="Folder of parsed document-tree jsonl files.")
    parser.add_argument("--output_folder", required=True, help="Folder for document trees with resolved links.")
    parser.add_argument("--index_file", required=True, help="Title index json produced from the parsed tree corpus.")
    parser.add_argument("--max_workers", type=int, default=8, help="Number of workers.")
    args = parser.parse_args()
    process_files(args.input_folder, args.output_folder, args.index_file, args.max_workers)


if __name__ == "__main__":
    main()
