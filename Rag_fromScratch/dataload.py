import PyPDF2


class pdfLoader:

    def __init__(self,pth):

        self.pth = pth
        self.textList = []
        self.text = ''

    def load(self):

        with open(self.pth,'rb') as pdf:

            reader = PyPDF2.PdfReader(pdf)
            total_pages = len(reader.pages)
            print(f"Total pages : {total_pages}")

            for page in reader.pages:
                self.textList.append(page.extract_text())

        self.text = "\n".join(self.textList)
        return self.text