
# InterstitialAlloys

The **Crystal Graph Network** pipeline is an open source platform that allows training of GNN models for prediction of properties of interstitial alloy-type materials.

In order to be able to run the experiments presented in the paper, you must create an environment using the yml file provided.

To reproduce all the experiments reported, just run the command:

<pre>
  <code>
    python train.py
  </code>
</pre>

To reproduce the results presented in supplementary information for Ti<sub>2</sub>C, run the command:

<pre>
  <code>
    python train.py --exp_name Ti2C_222 --max_d 2.5
  </code>
</pre>
