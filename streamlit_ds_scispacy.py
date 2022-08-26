# Core Pkgs
import streamlit as st 

# NLP Pkgs
import spacy_streamlit
import scispacy
import spacy
#nlp = spacy.load("en_core_sci_sm")
nlp = spacy.load("en_ner_bionlp13cg_md")

import os
#from PIL import Image

path = os.getcwd()
# Print the current working directory
# print("Current working directory: {0}".format(cwd))


def main():
    """A Simple NLP app with Spacy-Streamlit"""
    nweh_logo = Image.open(os.path.join('nweh_logo_sm.jpg')) 
    #st.image(nweh_logo)
    st.title("NER **for** **processing** **biomedical** **,** **scientific** **or** **clinical** **text** **with** _spaCy_ **NLP**")
    st.markdown('**_(_ _pre-trained_ _CNN_ _model_ _with_ _English_ _language_ _)_**')
    menu = ["Named Entity Recognision","Tokenization"]
    choice = st.sidebar.selectbox("Menu",menu)
    if choice == "Named Entity Recognision":
        st.subheader("Named Entity Recognision")
        raw_text = st.text_area("Sample Text","Myeloid derived suppressor cells (MDSC) are immature myeloid cells with immunosuppressive activity. They accumulate in tumor-bearing mice and humans with different types of cancer, including hepatocellular carcinoma (HCC).")
        docx = nlp(raw_text)
        if st.button("Analyse"):
	        spacy_streamlit.visualize_ner(docx,labels=nlp.get_pipe('ner').labels)

    elif choice == "Tokenization":
        st.subheader("Tokenization")
        raw_text = st.text_area("Sample Text","Myeloid derived suppressor cells (MDSC) are immature myeloid cells with immunosuppressive activity. They accumulate in tumor-bearing mice and humans with different types of cancer, including hepatocellular carcinoma (HCC).")
        docx = nlp(raw_text)
        if st.button("Analyse"):
            spacy_streamlit.visualize_tokens(docx,attrs=['text','pos_','dep_','ent_type_'])

    


if __name__ == '__main__':
	main()
