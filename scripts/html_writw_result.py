""" This script is used for writing in HTML files

It adds links to HTML table.
It generates span tags for un/colored matching blocks.
It compares two text files
It inserts comparison results in corresponding html files

"""

from os import fsync, path
from random import randint
from shutil import copyfile, copy
from typing import Any, List

from bs4 import BeautifulSoup as Bs
import importlib.resources
from tabulate import tabulate

from scripts.html_utils import (
    get_color_from_similarity,
    get_real_matching_blocks,
    blocks_list_to_strings_list,
    get_ordered_blocks_positions,
)
from scripts.utils import is_float




def add_links_to_html_table(html_path: str) -> None:
    """Add links to HTML data cells at specified path

    This method will link to all HTML TD tags which contain a float different from - 1 the
    corresponding HTML comparison file. The links will be opened in a new tab. The colors of
    the text in tag will change depending on similarity score.

    """

    with open(html_path, encoding="utf-8") as html:
        soup = Bs(html, "html.parser")
        file_ind = 0  # Cursor on file number for the naming of html files

        for td_tag in soup.findAll("td"):  # Retrieve all data celss from html table in path
            if is_float(td_tag.text):  # If td is not filename or -1
                tmp = soup.new_tag(
                    "a",
                    href="file:///" + html_path.replace("_results", str(file_ind)),
                    target="_blank",
                    style="color:" + get_color_from_similarity(float(td_tag.text)),
                )

                td_tag.string.wrap(tmp)  # We wrap the td string between the hyperlink
                file_ind += 1

        # We update the HTML of the file at path
        with open(html_path, "wb") as f_output:
            f_output.write(soup.prettify("utf-8"))
            f_output.flush()
            fsync(f_output.fileno())
            f_output.close()


def get_span_blocks(bs_obj: Bs, text1: list, text2: list, block_size: int) -> list:
    """Return list of spans with colors for HTML rendering"""

    results: List[List[Any]] = [[], []]  # List of spans list

    # Get matching blocks with chosen minimum size
    matching_blocks = get_real_matching_blocks(text1, text2, block_size)

    # Generate one unique color for each matching block
    colors = [f"#{randint(0, 0xFFFFFF):06X}" for _ in range(len(matching_blocks))]

    # Convert blocks from list of list of strings to list of strings
    string_blocks = [" ".join(map(str, text1[b.a : b.a + b.size])) for b in matching_blocks]

    # Store lengths of blocks in text
    strings_len_list = blocks_list_to_strings_list(matching_blocks, text1)

    # Convert list of strings to strings
    str1, str2 = " ".join(map(str, text1)), " ".join(map(str, text2))

    global_positions_list = [
        get_ordered_blocks_positions(str1, matching_blocks, string_blocks),
        get_ordered_blocks_positions(str2, matching_blocks, string_blocks),
    ]

    for num, pos_list in enumerate(global_positions_list):
        cursor = 0  # Cursor on current string

        if num == 1:  # Second iteration on second string
            str1 = str2

        for block in pos_list:
            # Span tag for the text before the matching sequence
            span = bs_obj.new_tag("span")
            span.string = str1[cursor : block[0]]

            # Span tag for the text in the matching sequence
            blockspan = bs_obj.new_tag("span", style="color:" + colors[block[1]] + "; font-weight:bold")
            blockspan.string = str1[block[0] : block[0] + strings_len_list[block[1]]]

            # Append spans tags to results list
            results[num].append(span)
            results[num].append(blockspan)

            # Update cursor position after last matching sequence
            cursor = block[0] + strings_len_list[block[1]]

        # End of loop, last span tag for the rest of the text
        span = bs_obj.new_tag("span")
        span.string = str1[cursor:]
        results[num].append(span)

    return results


def papers_comparison(save_dir: str, ind: int, text1: list, text2: list, filenames: tuple, block_size: int) -> None:
    """Write to HTML file texts that have been compared with highlighted similar blocks"""

    try:
        with importlib.resources.path("scripts", "template.html") as template_path:
            comp_path = path.join(save_dir, f"{ind}.html")
            copyfile(template_path, comp_path)
    except ModuleNotFoundError:
        # Fallback for local development
        template_path_local = path.join("template.html")
        comp_path = path.join(save_dir, str(ind) + ".html")

        # Copy the template to the save directory under a new name
        copy(template_path_local, comp_path)

    with open(comp_path, encoding="utf-8") as html:
        soup = Bs(html, "html.parser")
        res = get_span_blocks(soup, text1, text2, block_size)
        blocks = [soup.find(id="leftContent"), soup.find(id="rightContent")]

        # Append filename tags and span tags to html
        for i, filename in enumerate(filenames):
            temp_tag = soup.new_tag("h3")
            temp_tag.string = filename
            blocks[i].append(temp_tag)
            for tag in res[i]:
                blocks[i].append(tag)

    # Write the modified content back to the file
    with open(comp_path, "wb") as f_output:
        f_output.write(soup.prettify("utf-8"))



from typing import List
import os
from tabulate import tabulate
from os import fsync
from datetime import datetime

def generate_dynamic_css(scores: List[List[float]]) -> str:
    """
    Generate CSS for styling the similarity report page.
    """
    css = """
    <style>
    body {
        font-family: 'Arial', sans-serif;
        background-color: #f9f9f9;
        margin: 0;
        padding: 20px;
        color: #333;
    }
    .container {
        max-width: 1200px;
        margin: 0 auto;
        text-align: center;
    }
    h1 {
        font-size: 2.5em;
        color: #333;
        text-align: center;
        margin-bottom: 20px;
        text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.1);
    }
    .similarity-table {
        width: 100%;
        border-collapse: collapse;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
        background-color: white;
        margin-top: 20px;
    }
    .similarity-table th, .similarity-table td {
        border: 1px solid #ddd;
        padding: 12px;
        text-align: center;
        font-size: 1em;
        transition: background-color 0.3s ease, color 0.3s ease;
    }
    .similarity-table th {
        background-color: #0078D7;
        color: white;
        text-transform: uppercase;
    }
    .similarity-table tr:nth-child(even) {
        background-color: #f7f7f7;
    }
    .similarity-table tr:hover {
        background-color: #eaf4fb;
    }
    footer {
        margin-top: 20px;
        font-size: 0.9em;
        color: #555;
        text-align: center;
    }
    .tooltip {
        position: relative;
        display: inline-block;
        border-bottom: 1px dotted #555;
        font-size: 0.9em;
        color: #0078D7;
        cursor: help;
    }
    .tooltip .tooltiptext {
        visibility: hidden;
        width: 250px;
        background-color: #333;
        color: #fff;
        text-align: center;
        border-radius: 6px;
        padding: 8px;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        margin-left: -125px;
        opacity: 0;
        transition: opacity 0.3s;
    }
    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }
    </style>
    """
    return css


def results_to_html(scores: List[List[float]], files_names: List[str], html_path: str) -> None:
    """
    Write similarity results to an interactive HTML page
    
    Args:
        scores (List[List[float]]): Similarity scores matrix
        files_names (List[str]): Names of compared files
        html_path (str): Path to save the HTML file
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(html_path), exist_ok=True)

    # Ensure html_path is a full file path
    if not html_path.endswith('.html'):
        html_path = os.path.join(html_path, 'similarity_report.html')

    # Insert file names into scores matrix
    for ind, file_name in enumerate(files_names):
        scores[ind].insert(0, file_name)

    scores.insert(0, files_names)
    scores[0].insert(0, "Files")

    # Generate HTML table manually to avoid className issue
    html_table = tabulate(scores, tablefmt="html", headers="firstrow")
    
    # Replace the default table tag with our custom class
    html_table = html_table.replace('<table>', '<table class="similarity-table">')

    # Get current timestamp
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    current_user = os.getenv('USERNAME', 'Unknown User')

    # Generate HTML content
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Vervent Plagiarism</title>
        {generate_dynamic_css(scores)}
    </head>
    <body>
        <div class="container">
            <h1 style="font-size:3em; font-weight:bold; color:#0078D7; text-shadow: 2px 2px 5px rgba(0,0,0,0.2);">Vervent</h1>
            <h1>Plagiarism - Similarity Comparison Results</h1>
            <div class="tooltip">Hover over cells for details
                <span class="tooltiptext">Color intensity indicates similarity level</span>
            </div>
            {html_table}
        </div>
        
    </body>
    </html>
    """

    # Ensure the directory exists
    os.makedirs(os.path.dirname(html_path), exist_ok=True)

    with open(html_path, "w", encoding="utf-8") as file:
        file.write(html_content)
        file.flush()
        fsync(file.fileno())

    print(f"HTML report generated at {html_path}")
