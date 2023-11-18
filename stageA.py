import pdfplumber
from transformers import BertTokenizer, BertModel
import torch
from torch.nn.functional import cosine_similarity
import pandas as pd


def pdf2text(file_path: str):
    """
    Extract table from the given file path
    :param file_path: (string) the path of the pdf file
    :return: (list) A list of tables
    """
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:

            # Extract tables from the page
            tables = page.extract_tables()

    return tables


class text2vector():
    def __init__(self):
        # Load pre-trained model and tokenizer
        model_name = "bert-base-chinese"
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)

    def to_vector(self, input_sentence: str):
        """
        Convert text to string
        Input: (string) A sentence
        Output: sentence vector
        """
        # Tokenize and encode sentences
        token = self.tokenizer(input_sentence, return_tensors="pt")

        # Generate embeddings
        with torch.no_grad():
            token_embd = self.model(**token)

        # Use the mean of the last hidden states as sentence vector
        vector = token_embd.last_hidden_state.mean(dim=1)

        return vector


def cosine_sim(input1: str, input2: str):
    """
    Comparing the cosine similarity from the two str
    :param input1: (str)
    :param input2: (str)
    :return: (float) cosine similarity
    """
    t2v = text2vector()

    vec1, vec2 = t2v.to_vector(input1), t2v.to_vector(input2)
    similarity = cosine_similarity(vec1, vec2)

    return similarity


def main(keyword, pdf_file):
    # get the tables from pdf file
    tables = pdf2text(pdf_file)

    # transform the table and compare the cosine similarity
    max_cos_sim = 0
    max_cos_sim_table = -1
    for n, table in enumerate(tables):

        num_rows = len(table)
        num_columns = len(table[0])

        # make the element of the table a sentence
        for i in range(1, num_rows):

            for j in range(1, num_columns):
                table_item = tables[n][i][j].split("\n")
                tables[n][i][j] = ''.join(table_item)

                table_sentence = table[0][j] + "的" + table[i][0]
                similarity = cosine_sim(table_sentence, keyword)

                if similarity > max_cos_sim:
                    max_cos_sim_table = n

    assert max_cos_sim_table != -1, "Something is wrong"

    return tables[max_cos_sim_table]


if __name__ == "__main__":
    # UI interfaces
    pdf_path = input("PDF file path:")
    keyword = input("Searching Keywords:")
    table = main(keyword, pdf_path)
    df = pd.DataFrame(table)
    print(df)
