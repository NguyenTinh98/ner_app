from flask import Flask,url_for,render_template,request
import json
import utils
import requests


HTML_WRAPPER = """<div style="overflow-x: auto; border: 1px solid #e6e9ef; border-radius: 0.25rem; padding: 1rem">{}</div>"""

# my_url  = 'http://192.168.1.9:3002/api/'
from flaskext.markdown import Markdown

app = Flask(__name__)
Markdown(app)


# def analyze_text(text):
# 	return nlp(text)

@app.route('/')
def index():
	# raw_text = "Bill Gates is An American Computer Scientist since 1986"
	# docx = nlp(raw_text)
	# html = displacy.render(docx,style="ent")
	# html = html.replace("\n\n","\n")
	# result = HTML_WRAPPER.format(html)
	return render_template('result.html', raw_text = '', result='')


@app.route('/',methods=["GET","POST"])
def extract():
    if request.method == 'POST':
        if request.form.get('submit_button', False) == "NE Recognition":
            raw_text = request.form['rawtext']
            # docx = ner.predict(raw_text)
            # r = requests.post(url= my_url, data={'text':raw_text})
            docx = json.loads(r.text)
            html = utils.visualize_spacy(docx)
            html = html.replace('[/n]','</br>')
            result = HTML_WRAPPER.format(html)
        elif request.form.get('clear_button', False) == "Clear":
            raw_text = ''
            result = ''
    return render_template('result.html', raw_text=raw_text, result=result)


if __name__ == '__main__':
    app.run(port = 1612, debug=True)
