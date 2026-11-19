# Literature Review

This review covers the work that the project builds on. It is organized by question, not by date.

## 1. Random labels as a diagnostic

Zhang et al. (2017) train networks on CIFAR-10 and ImageNet after corrupting the data. In one setting, a fraction p of labels is replaced with uniformly random classes. Networks fit all corrupted versions to zero training error, and test error rises toward chance as corruption grows. The same paper shows that weight decay, dropout and data augmentation do not stop random-label fitting. A later retrospective by the same authors (Zhang et al., 2021) restates these findings and reviews follow-up theory.

Arpit et al. (2017) refine this picture. Networks learn real patterns before they memorize noise, and regularization slows memorization more than it slows learning. Rolnick et al. (2017) find that deep networks still reach high test accuracy when mislabeled examples greatly outnumber clean ones, as long as enough clean data remain.

**Relevance to this project:** these papers motivate the question we ask on tabular data. A low training error on noisy labels does not show that a model generalizes. We therefore track training accuracy on the noisy labels next to clean-label test performance.

## 2. Label noise and robust learning

Angluin and Laird (1988) study learning from noisy examples. Under symmetric label noise, the Bayes-optimal decision rule does not change. Natarajan et al. (2013) show that some loss functions are robust to this kind of noise. Frénay and Verleysen (2014) survey the wider literature on classification with label noise.

**Relevance to this project:** we do not propose a noise-correction method. We measure how standard, unmodified estimators respond to increasing label randomization.

## 3. Model families under noise

- **Random forests.** Breiman (2001) introduces random forests as averages of randomized decision trees. Fully grown trees interpolate their training data. Wyner et al. (2017) argue that such interpolating ensembles can still generalize, because averaging localizes the effect of noisy points. Grinsztajn et al. (2022) find that tree-based models still outperform deep learning on typical tabular data.
- **Multilayer perceptrons.** The MLP is trained with Adam (Kingma and Ba, 2015).
- **Logistic regression** provides a regularized linear rule after standardization.
- All estimators are implemented in scikit-learn (Pedregosa et al., 2011).

## 4. The dataset

Koklu and Ozkan (2020) introduce the Dry Bean dataset and classify seven varieties from shape features computed on bean images. They report cross-validated accuracy on the full dataset. Our study deduplicates the data, uses a fixed split and reports Macro-F1, so their numbers are not directly comparable to ours.

## Where this project fits

The random-label experiments above use image benchmarks and deep networks. This project asks a narrower question on one tabular dataset: how LR, RF and an MLP with standard settings behave as training labels are randomized, over five noise levels and three seeds.

## References

- Angluin, D., and Laird, P. (1988). Learning from noisy examples. *Machine Learning*, 2(4), 343–370. https://doi.org/10.1007/BF00116829
- Arpit, D., et al. (2017). A closer look at memorization in deep networks. *ICML 2017*, PMLR 70, 233–242. https://proceedings.mlr.press/v70/arpit17a.html
- Breiman, L. (2001). Random forests. *Machine Learning*, 45, 5–32. https://doi.org/10.1023/A:1010933404324
- Frénay, B., and Verleysen, M. (2014). Classification in the presence of label noise: A survey. *IEEE TNNLS*, 25(5), 845–869. https://doi.org/10.1109/TNNLS.2013.2292894
- Grinsztajn, L., Oyallon, E., and Varoquaux, G. (2022). Why do tree-based models still outperform deep learning on typical tabular data? *NeurIPS 2022, Datasets and Benchmarks Track*.
- Kingma, D. P., and Ba, J. (2015). Adam: A method for stochastic optimization. *ICLR 2015*. https://arxiv.org/abs/1412.6980
- Koklu, M., and Ozkan, I. A. (2020). Multiclass classification of dry beans using computer vision and machine learning techniques. *Computers and Electronics in Agriculture*, 174, 105507. https://doi.org/10.1016/j.compag.2020.105507
- Natarajan, N., et al. (2013). Learning with noisy labels. *NeurIPS 26*.
- Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *JMLR*, 12, 2825–2830. https://www.jmlr.org/papers/v12/pedregosa11a.html
- Rolnick, D., et al. (2017). Deep learning is robust to massive label noise. arXiv:1705.10694.
- Wyner, A. J., Olson, M., Bleich, J., and Mease, D. (2017). Explaining the success of AdaBoost and random forests as interpolating classifiers. *JMLR*, 18(48), 1–33.
- Zhang, C., Bengio, S., Hardt, M., Recht, B., and Vinyals, O. (2017). Understanding deep learning requires rethinking generalization. *ICLR 2017*. https://arxiv.org/abs/1611.03530
- Zhang, C., et al. (2021). Understanding deep learning (still) requires rethinking generalization. *Communications of the ACM*, 64(3), 107–115. https://doi.org/10.1145/3446776
