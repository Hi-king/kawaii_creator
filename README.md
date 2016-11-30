# kawaii_creator

機械学習で美少女化 

使い方
---

まずモデルをダウンロード

```
wget https://www.dropbox.com/s/wm18gzjbx5359eq/generator_model_4050000.npz
wget https://www.dropbox.com/s/x83ys1gxxxoo6og/vectorizer_model_4050000.npz
```

###  画像を変換

```
python bin/vectorize.py generator_model_4050000.npz vectorizer_model_4050000.npz input.jpg  --out_file=output.png --real_face --show
```

###  カメラからのリアルタイム変換

```
python bin/face2d.py generator_model_4050000.npz vectorizer_model_4050000.npz
```

![myface2d](sample/myface2d.gif)
