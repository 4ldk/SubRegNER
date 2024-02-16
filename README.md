# Subword Regularization For NER(Conll2023)

論文:[サブワード系列の変化が固有表現抽出に与える影響の調査]()

このリポジトリは「サブワード系列の変化が固有表現抽出に与える影響の調査」で使用したスクリプトです。BERT/RoBERTa/LUKEに対して、サブワード正則化(MaxMatch-Dropout/BPE-Dropout/BPE-Dropout)やMulti-viewサブワード正則化を適用してNERの学習/推論を行えます(保守を行っていないため、動かないものがある可能性があります)。

データセットはCoNLL2003のeng.train, eng.testa, eng.tesbに加えてCoNLL++(2023)のconllpp.txt、CoNLL++(CrossWeight)のconllpp_test.txtをconllcw.txtとファイル名を変更したものをすべてtest_datasetsフォルダ内に入れることでそれぞれ利用できます。

config/conll2003.yamlを編集後、各モデルのxxx_finetuning.pyを実行すると学習、xxx_pred_roop.pyで推論を行います。
また、run.pyでfinetuningからpredictまでを自動で行います。(LUKEのみfinetuningを用意していません。LUKE_pred.pyで推論のみができます。また、LUKEのみ出力フォーマットが他と少し違います。)

analyzeフォルダ内のファイルを実行する場合、各モデルの推論で得られた推論結果のtxtファイルをoutput100/(モデル名)/(学習設定).txt(例.RoBERTa Baseでsubword正則化のハイパーパラメータp=0.1で学習した場合: output100/RoBERTaB/Reg.txt)にすることで使用できます。(LUKEではluke/100pred_analysis.pyを利用してください)

test datasets
- CoNLL2003: https://www.clips.uantwerpen.be/conll2003/ner/
- CoNLL++(2023): https://github.com/ShuhengL/acl2023_conllpp
- CoNLL++(CrosWeigh): https://github.com/ZihanWangKi/CrossWeigh

学習設定
- MaxMatchTokenizer: https://github.com/tatHi/maxmatch_dropout
- BPE-Dropout: https://aclanthology.org/2020.acl-main.170/
- Muti-view Subword Regularizaton: https://aclanthology.org/2021.naacl-main.40/

リンクがgithubのものはそのコード/データをそのまま利用したもの、論文のものは自前で実装したものです。正確な参考文献は論文を参照してください。
