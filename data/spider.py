
import time
import random
import requests
from pypinyin import pinyin,Style
import urllib.parse

def download(url, name):
    resp = requests.get(url)
    if resp.status_code != 200:
        with open("error.log", 'a') as w:
            w.write(str(resp)+"\t"+url)
        time.sleep(10)
        return
    else:
        with open(name, "wb") as writer:
            writer.write(resp.content)
            writer.close()

def work():
    for line in open('pinyin.data', 'r').readlines():
        name = line.strip() + ".mp3"
        print("downloading %s" % name)
        download("https://fanyiapp.cdn.bcebos.com/zhdict/mp3/%s" % name, name)
        time.sleep(2+random.random())

def work2():
    for line in open('words', "r").readlines():
        if len(line) > 5:
            continue
        py = pinyin(line.strip(), style=Style.TONE3, heteronym=False)
        text = ""
        index = 0
        name = []
        for ch in line.strip():
            text += ch
            text += '(%s)' % py[index][0]
            name.append(py[index][0])
            index += 1
        print("downloading %s" % text)
        download("https://tts.baidu.com/text2audio?tex=%s&cuid=dict&lan=ZH&ctp=1&pdt=30&vol=9&spd=4" % urllib.parse.quote(text), "|".join(name)+".mp3")
        time.sleep(2+random.random())

if __name__ == "__main__":
    work2()
    #download("https://fanyiapp.cdn.bcebos.com/zhdict/mp3/quan2.mp3", "quan2.mp3")
