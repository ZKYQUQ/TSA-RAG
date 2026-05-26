import json
import regex as re
import os
import logging
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import copy
import argparse

import xml.etree.ElementTree as ET

def parse_wikipedia_dump(xml_file, output_jsonl):
    with open(xml_file, 'r', encoding='utf-8') as file, open(output_jsonl, 'w', encoding='utf-8') as outfile:
        context = ET.iterparse(file, events=('end',))
        
        for event, elem in context:
            if elem.tag == '{http://www.mediawiki.org/xml/export-0.10/}page':
                title = elem.find('{http://www.mediawiki.org/xml/export-0.10/}title').text
                revisions = elem.findall('{http://www.mediawiki.org/xml/export-0.10/}revision')
                latest_revision = revisions[-1] if revisions else None
                
                if latest_revision is not None:
                    text_elem = latest_revision.find('{http://www.mediawiki.org/xml/export-0.10/}text')
                    text = text_elem.text if text_elem is not None else ""
                    
                    json_record = json.dumps({'title': title, 'text': text}, ensure_ascii=False)
                    outfile.write(json_record + '\n')
                
                elem.clear()


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

    # remove link brackets, match whole content of the second part, e.g.
    # [[Niassa Province]], [[32 Battalion (South Africa)|32 Battalion]], [[2004 Wimbledon Championships|2004]]
    # [[Operation Savannah (Angola)#&quot;Bridge 14&quot;|Battle of Bridge 14]]
    text = re.sub(r'\[\[([^|\]]+\|)?([^\]]+)\]\]', r'\2', text)

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


def extract_text(title, text, list_mode=True):
    result = []
    id_counter = 0
    node_index = 0

    # create root node
    root_node = {
        "id": id_counter,
        "text": title,
        "type": "title",
        "relation": {
            "up_id": -1,
            "down_ids": []
        },
        "span": [node_index, node_index + len(title)]
    }
    result.append(root_node)
    id_counter += 1
    # consider \n add 1
    node_index += len(title) + 1

    # initialize title stack
    top_node_level = 1
    top_node = root_node
    stack = [(top_node, top_node_level)]

    parts = text.split('\n')
    # consider redundant title and its subordinate parts, e.g.
    # ==External links==    links
    # == Notes and references ==, ==References==    ref info
    # ===Notes===, ==Footnotes==    links or ref info
    # ==See also==    links
    # ==Sources==    links
    skip_contain_titles = ["external links", "notes", "references", "see also", "gallery", "images", "pictures", "photographs", "sources", "citations"]
    # some may be useful    Open data: ==Major sources==
    skip_equal_titles = []
    skip_mode = False
    skip_level = None

    # * list
    # first level: *[[Balconies]]    * The overall winner,
    # second level: **'''Note''' - I

    list_stack = []
    list_parent_node = None

    for part in parts:
        if not part:
            continue

        # skip image files
        # [[File:Villa Fridheim, Krøderen...]]
        # File:Maurice Cullen
        if part.lower().startswith("[[file:") or part.lower().startswith("file:"):
            continue

        match_title = re.match(r'(={2,})\s*(.*?)\s*\1', part)
        if list_mode:
            match_list = re.match(r'(:*)([\*#]+)\s*(.*)', part)
        else:
            match_list = None
        # the part is a title
        if match_title:
            level = len(match_title.group(1))
            raw_text = match_title.group(2).strip()
            heading_text = '=' * level + raw_text + '=' * level

            # meet new title of the same or higher level, exit skip mode
            if skip_mode and level <= skip_level:
                skip_mode = False
                skip_level = None

            # meet subordinate parts of external links
            if skip_mode:
                continue

            # detect redundant section and start skipping
            if any(keyword in raw_text.lower() for keyword in skip_contain_titles):
                skip_mode = True
                skip_level = level
                continue

            if any(keyword == raw_text.lower() for keyword in skip_equal_titles):
                skip_mode = True
                skip_level = level
                continue

            # meet new title of the same or higher level, pop stack
            while level <= stack[-1][1]:
                stack.pop()

            top_node = stack[-1][0]

            new_node = {
                "id": id_counter,
                "text": heading_text,
                "type": "title",
                "relation": {
                    "up_id": top_node["id"],
                    "down_ids": []
                },
                "span": [node_index, node_index + len(heading_text)]
            }
            result.append(new_node)
            id_counter += 1
            # consider \n add 1
            node_index += len(heading_text) + 1

            top_node["relation"]["down_ids"].append(new_node["id"])

            top_node = new_node
            stack.append((top_node, level))
            list_stack.clear()
            list_parent_node = None

        # the part is a list item
        elif match_list:
            if skip_mode:
                continue

            indent_level = len(match_list.group(1))
            list_type = match_list.group(2)[0]
            item_text = match_list.group(1) + match_list.group(2) + " " + match_list.group(3).strip()
            list_level = len(match_list.group(2)) + indent_level

            if not match_list.group(3).strip():
                continue

            if not list_stack:
                list_parent_node = result[-1]

            while list_stack and list_stack[-1][1] >= list_level:
                list_stack.pop()

            if list_stack:
                parent_node = list_stack[-1][0]
            else:
                parent_node = list_parent_node

            new_node = {
                "id": id_counter,
                "text": item_text,
                "type": "content",
                "relation": {
                    "up_id": parent_node["id"],
                    "down_ids": []
                },
                "span": [node_index, node_index + len(item_text)]
            }
            result.append(new_node)
            id_counter += 1
            # consider \n add 1
            node_index += len(item_text) + 1

            parent_node["relation"]["down_ids"].append(new_node["id"])
            list_stack.append((new_node, list_level))

        # the part is a paragraph
        else:
            if skip_mode:
                continue

            if not part.strip():
                continue

            if part.strip():
                new_node = {
                    "id": id_counter,
                    "text": part.strip(),
                    "type": "content",
                    "relation": {
                        "up_id": top_node["id"],
                        "down_ids": []
                    },
                    "span": [node_index, node_index + len(part.strip())]
                }
                result.append(new_node)
                id_counter += 1
                # consider \n add 1
                node_index += len(part.strip()) + 1

                top_node["relation"]["down_ids"].append(new_node["id"])
                list_stack.clear()
                list_parent_node = None

    return result

    


def parse_to_jsonl(jsonl_file, output_jsonl_file, list_mode=True, remove_empty_titles = True):
    # filter useless pages via title
    # Wikipedia:Articles for deletion/Nikhil Parekh
    # Wikipedia:WikiProject British crime/Navigation
    # Wikipedia:Editor review/Miller17CU94
    # Category:Wikipedians who like Yakitate!! Japan
    # Template:Ukroadsmall
    # File:AnneMcCaffrey Dragonflight.jpg
    useless_title_keywords = ("wikipedia:", "category:", "template:", "file:")

    with open(jsonl_file, 'r', encoding='utf-8') as f, open(output_jsonl_file, 'w', encoding='utf-8') as out_f:

        for line in f:
            data = json.loads(line)
            title = data.get('title')
            text = data.get('text')
            if title == "Stability and Growth Pact" or title == "ROH World Television Championship":
                continue

            if title is None or text is None:
                continue

            # filter ambiguous pages via title
            if "(disambiguation)" in title.lower() or "(disambiguation page)" in title.lower():
                continue

            # filter useless pages via title
            if any(title_keyword in title.lower() for title_keyword in useless_title_keywords):
                continue

            # # Take out List/Index/Outline pages (mostly links)
            # # e.g. list of tales, festivals
            # List of technology centers
            if re.match(r"(List of .+)|(Index of .+)|(Outline of .+)", title):
                continue

            # filter redirect pages via text
            if text.startswith("#REDIRECT") or text.startswith("#redirect") or text.startswith("#Redirect"):
                continue

            cleaned_text = clean_text(text)
            if cleaned_text is None:
                continue

            result = extract_text(title, cleaned_text, list_mode)
            full_text = "\n".join(node["text"] for node in result)
            num_nodes = len(result)

            # remove empty title nodes and reassign node ids
            if remove_empty_titles:
                ## assign down_ids of empty titles to be empty
                copy_result = copy.deepcopy(result)
                tmp_node = None
                for node in copy_result:
                    if node["type"] == "title" and not node["relation"]["down_ids"]:
                        tmp_node = node
                        while not tmp_node["relation"]["down_ids"]:
                            parent_node_id = tmp_node["relation"]["up_id"]
                            if parent_node_id >= 0:
                                copy_result[parent_node_id]["relation"]["down_ids"].remove(tmp_node["id"])
                            else:
                                break
                            tmp_node = copy_result[parent_node_id]
                ## reassign ids of nodes other than empty titles
                result_without_empty_title = []
                node_count = 0
                for node in copy_result:
                    if node["type"] == "title" and not node["relation"]["down_ids"]:
                        continue
                    node_count += 1
                
                for node in reversed(copy_result):
                    if node["type"] == "title" and not node["relation"]["down_ids"]:
                        continue

                    node["id"] = node_count - 1
                    node_count -= 1
                    down_ids = node["relation"]["down_ids"]
                    for down_id in down_ids:
                        index = down_ids.index(down_id)
                        down_ids[index] = copy_result[down_id]["id"]
                        copy_result[down_id]["relation"]["up_id"] = node["id"]
                
                ## insert nodes into new list 
                for node in reversed(copy_result):
                    if node["type"] == "title" and not node["relation"]["down_ids"]:
                        continue
                    result_without_empty_title.insert(0, node)

                if not result_without_empty_title:
                    result_without_empty_title.append(result[0])
                    result_without_empty_title[0]["relation"]["down_ids"].clear()

                # remove title words in full_text and reassign the spans of content nodes 
                tmp_index = 0
                tmp_index += len(title) + 1
                text_without_title = title

                for node in result_without_empty_title[1:]:
                    if node["type"] == "content":
                        node["span"][0] = tmp_index
                        node["span"][1] = tmp_index + len(node["text"])
                        tmp_index += len(node["text"]) + 1
                        text_without_title += "\n" + node["text"]
                    else:
                        # node["span"] = []
                        node["span"].clear()

                output_data = {
                    "title": title,
                    "num_nodes": len(result_without_empty_title),
                    "extracted_nodes": result_without_empty_title,
                    "full_text": text_without_title,
                    "raw_text": text
                }
            else:
                # remove title words in full_text and reassign the spans of content nodes 
                tmp_index = 0
                tmp_index += len(title) + 1
                text_without_title = title

                for node in result[1:]:
                    if node["type"] == "content":
                        node["span"][0] = tmp_index
                        node["span"][1] = tmp_index + len(node["text"])
                        tmp_index += len(node["text"]) + 1
                        text_without_title += "\n" + node["text"]
                    else:
                        # node["span"] = []
                        node["span"].clear()

                output_data = {
                    "title": title,
                    "num_nodes": num_nodes,
                    "extracted_nodes": result,
                    "full_text": text_without_title,
                    "raw_text": text
                }

            out_f.write(json.dumps(output_data, ensure_ascii=False) + '\n')


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def process_single_file(xml_file, temp_folder, output_folder):
    try:
        filename = os.path.basename(xml_file)
        output_jsonl_file = os.path.join(output_folder, filename.replace('.xml', '.jsonl'))
        intermediate_jsonl = os.path.join(temp_folder, f"{filename}.temp.jsonl")

        logging.info(f"Processing {xml_file} -> {intermediate_jsonl}")
        parse_wikipedia_dump(xml_file, intermediate_jsonl)

        logging.info(f"Processing {intermediate_jsonl} -> {output_jsonl_file}")
        parse_to_jsonl(intermediate_jsonl, output_jsonl_file, True, True)
        logging.info(f"===Successfully process {output_jsonl_file} ===")

    except Exception as e:
        logging.error(f"Error processing {xml_file}: {e}")
        logging.error(f"Error details: {str(e)}", exc_info=True)

def process_files(input_folder, temp_folder, output_folder, max_workers=None):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    if not os.path.exists(temp_folder):
        os.makedirs(temp_folder)

    xml_files = [os.path.join(input_folder, filename) for filename in os.listdir(input_folder) if '.xml' in filename]

    if max_workers is None:
        max_workers = cpu_count()
    logging.info(f"Using {max_workers} workers for processing.")

    with Pool(processes=max_workers) as pool:
        list(tqdm(
            pool.starmap(process_single_file, [(xml_file, temp_folder, output_folder) for xml_file in xml_files]),
            total=len(xml_files),
            desc="Processing files",
            unit="file"
        ))

    logging.info("All files processed successfully.")

def main():
    parser = argparse.ArgumentParser(description="Parse Wikipedia XML dumps into TSA-RAG document-tree jsonl files.")
    parser.add_argument("--input_folder", required=True, help="Folder containing Wikipedia XML dump shards.")
    parser.add_argument("--temp_folder", required=True, help="Temporary folder for intermediate plain jsonl pages.")
    parser.add_argument("--output_folder", required=True, help="Output folder for parsed tree jsonl files.")
    parser.add_argument("--max_workers", type=int, default=8, help="Number of parser workers.")
    args = parser.parse_args()
    process_files(args.input_folder, args.temp_folder, args.output_folder, args.max_workers)


if __name__ == "__main__":
    main()
