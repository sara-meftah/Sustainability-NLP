# Tracking SOTA papers about Neural Natural Langauge Processing for Climate and Sustainability applications

> We only consider works from 2020

## [Annotated Datasets](#content)

### [Climate & Environment](#content)

 - **ClimaText: A Dataset for Climate Change Topic Detection** - 2020 - arxiv [paper](https://arxiv.org/abs/2012.00483) [website](https://www.sustainablefinance.uzh.ch/en/research/climate-fever/climatext.html)
 - **CLIMATE-FEVER: A Dataset for Verification of Real-World Climate Claims** - 2020 - arxiv [paper](https://arxiv.org/pdf/2012.00614.pdf) [website](https://www.sustainablefinance.uzh.ch/en/research/climate-fever.html)
- **A Dataset for Detecting Real-World Environmental Claims** - 2022 - arxiv [paper](https://arxiv.org/pdf/2209.00507.pdf) [website](https://github.com/dominiksinsaarland/environmental_claims)
- **Towards Fine-grained Classification of Climate Change related Social Media Text** - 2022 - Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics: Student Research Workshop [paper](https://aclanthology.org/2022.acl-srw.35.pdf) [website](https://github.com/roopalv54/finegrained-climate-change-social-media)
- **ClimaBench: A Benchmark Dataset For Climate Change Text Understanding in English** - 2023 - arxiv [paper](https://arxiv.org/pdf/2301.04253.pdf) [website](https://github.com/climabench/climabench) **The datasets are not publicly available** 
<!--- - **XX** - 2022 - arxiv [paper]() [website]()-->

### [ESG](#content)

- **FinSim4-ESG Shared Task: Learning Semantic Similarities for the Financial Domain. Extended edition to ESG insights** - 2022 - [paper](https://aclanthology.org/2022.finnlp-1.28.pdf) [website](https://sites.google.com/nlg.csie.ntu.edu.tw/finnlp-2022/home) **The datasets are not publicly available** 
<!---- **XX** - 2022 - arxiv [paper]() [website]()-->




## [Pretrained models](#content)

### [LLMs](#content)

- **ClimateBERT:** "Using the DistilRoBERTa model as starting point, the climatebert/distilroberta-base-climate-f Language Model is additionally pre-trained on a text corpus comprising climate-related research paper abstracts, corporate and general news and reports from companies."  - 2022 - [paper](https://arxiv.org/pdf/2110.12010.pdf) [model](https://huggingface.co/climatebert/distilroberta-base-climate-f)
- **ESG-BERT:** "Using the Google’s “BERT” language model as starting point, ESG-BERT is further fine-tuned on large unstructured Sustainability text corpora."  - 2020 - [paper](https://arxiv.org/pdf/2203.16788.pdf) [description](https://towardsdatascience.com/nlp-meets-sustainable-investing-d0542b3c264b) [model](https://huggingface.co/nbroad/ESG-BERT)
- **climateGPT-2** "GPT-based models pre-trained on climate change corpora, consisting of over 360 thousand abstracts of top climate scientists’ articles from trustable sources covering large temporal and spatial scales" - 2022 - [paper](https://s3.us-east-1.amazonaws.com/climate-change-ai/papers/neurips2022/27/paper.pdf) [model- in progress - for more details](https://github.com/huggingface/transformers/issues/20747)
- **FinBERT pretrain** "The pretrained FinBERT model on large-scale financial texts." - 2022 - [paper](https://arxiv.org/pdf/2006.08097.pdf) [github](https://github.com/yya518/FinBERT) [model](https://huggingface.co/yiyanghkust/finbert-pretrain)

### [Classification](#content)

- **environmental-claims:** "The environmental-claims model is fine-tuned on the EnvironmentalClaims dataset by using the climatebert/distilroberta-base-climate-f model as pre-trained language model." - 2022 - [paper](https://arxiv.org/pdf/2209.00507.pdf) [model](https://huggingface.co/climatebert/environmental-claims)
- **climatebert-fact-checking:** "This model fine-tuned ClimateBert on the textual entailment task using Climate FEVER data. Given (claim, evidence) pairs, the model predicts support (entailment), refute (contradict), or not enough info (neutral)." - 2022 - [model](https://huggingface.co/amandakonet/climatebert-fact-checking)
- **climabench/miniLM-cdp-all** - 2023 - [paper](https://arxiv.org/pdf/2301.04253.pdf) [model](https://huggingface.co/climabench/miniLM-cdp-all)) **More details are needed**
- **FinBERT ESG** "FinBERT-ESG is a FinBERT model fine-tuned on 2,000 manually annotated sentences from firms' ESG reports and annual reports (Environmental, Social, Governance or None)." - 2022 - [paper](https://arxiv.org/pdf/2006.08097.pdf) [github](https://github.com/yya518/FinBERT) [model](https://huggingface.co/yiyanghkust/finbert-esg)
- **FinBERT ESG 9 categories** "FinBERT-esg-9-categories is a FinBERT model fine-tuned on about 14,000 manually annotated sentences from firms' ESG reports and annual reports. finbert-esg-9-categories classifies a text into nine fine-grained ESG topics: Climate Change, Natural Capital, Pollution & Waste, Human Capital, Product Liability, Community Relations, Corporate Governance, Business Ethics & Values, and Non-ESG. This model complements finbert-esg which classifies a text into four coarse-grained ESG themes (E, S, G or None)." - 2022 - [paper](https://arxiv.org/pdf/2006.08097.pdf) [github](https://github.com/yya518/FinBERT) [model](https://huggingface.co/yiyanghkust/finbert-esg-9-categories) [description](https://www.allenhuang.org/uploads/2/6/5/5/26555246/esg_9-class_descriptions.pdf)


### [QA](#content)

- **ClimateBERTqa:** "This model is a fine-tuned version of climatebert/distilroberta-base-climate-f on the squad_v2 dataset." - 2022 - [model](https://huggingface.co/NinaErlacher/ClimateBERTqa)


### [Sentiment Analysis](#content)

- **FinBERT Sentiment** "This released finbert-tone model is the FinBERT model fine-tuned on 10,000 manually annotated (positive, negative, neutral) sentences from analyst reports." - 2022 - [paper](https://arxiv.org/pdf/2006.08097.pdf) [github](https://github.com/yya518/FinBERT) [model](https://huggingface.co/yiyanghkust/finbert-tone)



## [Papers](#content)

### [2023](#content)

-  **Machine Learning Methods in Climate Finance: A Systematic Review** - Documentos de Trabajo/Banco de España - 2023 - [paper](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4352569) [github - not available]()
-  **Evaluating TCFD Reporting: A New Application of Zero-Shot Analysis to Climate-Related Financial Disclosures** - arxiv - 2023 - [paper](https://arxiv.org/abs/2302.00326) [github - not available]()
-  **ClimaBench: A Benchmark Dataset For Climate Change Text Understanding in English** - arxiv - 2023 - [paper](https://arxiv.org/abs/2301.04253) [github](https://github.com/climabench/climabench)
-  **Financial Language Understandability Enhancement Toolkit** - CODS-COMAD '23: Proceedings of the 6th Joint International Conference on Data Science & Management of Data  - 2023 - [paper](https://dl.acm.org/doi/abs/10.1145/3570991.3571067) [code - colab](https://colab.research.google.com/drive/1-KBBKByCU2bkyAUDwW-h6QCSqWI8z127?usp=sharing) [model](https://huggingface.co/spaces/sohomghosh/FLUEnT)
-  **Using Natural Language Processing to Enhance Understandability of Financial Texts** - CODS-COMAD '23: Proceedings of the 6th Joint International Conference on Data Science & Management of Data  - 2023 - [paper](https://dl.acm.org/doi/abs/10.1145/3570991.3571051) [github - not available]()
-  **ESG information extraction with cross-sectoral and multi-source adaptation based on domain-tuned language models** - Expert Systems with Applications - ELSEVIER - 2023 - [paper](https://www.sciencedirect.com/science/article/abs/pii/S0957417423002270) [github - not available]()
-  **Evaluating TCFD Reporting: A New Application of Zero-Shot Analysis to Climate-Related Financial Disclosures** - arxiv - 2023 - [paper](https://arxiv.org/abs/2302.00326) [github - not available]()
-  **Greenwashing, Sustainability Reporting, and Artificial Intelligence: A Systematic Literature Review** - MDPI Sustainability - 2023 - [paper](https://www.mdpi.com/2071-1050/15/2/1481) [github - not available]()
-  **XX** - XXX - 2023 - [paper]() [github - not available]()

### [2022](#content)

-  **Different Shades of Green: Estimating the Green Bond Premium using Natural Language Processing** - Swiss Finance Institute Research Paper - 2022 - [paper](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4198065) [github - not available]()
-  **Cheap Talk in Corporate Climate Commitments: The Role of Active Institutional Ownership, Signaling, Materiality, and Sentiment** - Swiss Finance Institute Research Paper - 2022 - [paper](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4000708) [github - not available]()
-  **Let’s get physical: Comparing metrics of physical climate risk** - Finance Research Letters - 2022 - [paper](https://www.sciencedirect.com/science/article/pii/S1544612321004013)  [github - not available]()
-  **Cheap talk and cherry-picking: What ClimateBert has to say on corporate climate risk disclosures** - Finance Research Letters - 2022 - [paper](https://www.sciencedirect.com/science/article/pii/S1544612322000897)  [github - not available]()
-  **Cheap Talk in Corporate Climate Commitments: The effectiveness of climate initiatives** - Swiss Finance Institute Research Paper Series - 2022 - [paper](https://papers.ssrn.com/sol3/Papers.cfm?abstract_id=3998435) [github - not available]()
-  **Thus spoke GPT-3: Interviewing a large-language model on climate finance** - Finance Research Letters - 2022 - [paper](https://www.sciencedirect.com/science/article/pii/S1544612322007930)  [github - not available]()
-  **Green Investors and Green Transition Efforts: Talk the Talk or Walk the Walk?** -   - 2022 - [paper](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4254894)  [github - not available]()
-  **Consistent and replicable estimation of bilateral climate finance** - nature climate change - 2022 - [paper](https://www.nature.com/articles/s41558-022-01482-7)  [github - not available]()
-  **Using Text Classification with a Bayesian Correction for Estimating Overreporting in the Creditor Reporting System on Climate Adaptation Finance.** - arxiv - 2022 - [paper](https://arxiv.org/abs/2211.16947)  [github - not available]()
-  **Mandatory CSR reporting in Europe: A textual analysis of firms’ climate disclosure practices** -  - 2022 - [paper](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4231567)   [github - not available]()
-  **MICRO-DATABASE FOR SUSTAINABILITY (ESG) INDICATORS** -   - 2022 - [paper](https://repositorio.bde.es/bitstream/123456789/29496/1/nest17.pdf)  [github - not available]()
-  **Climatebug: A Data-Driven Framework for Analyzing Bank Reporting Through a Climate Lens** -  - 2022 - [paper](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4308287)  [github - not available]()
-  **Tackling climate change with machine learning** - ACM Computing Surveys - 2022 - [paper](https://dl.acm.org/doi/full/10.1145/3485128)  [github - not available]()
-  **Analyzing and Visualizing Text Information in Corporate Sustainability Reports Using Natural Language Processing Methods** - Applied Sciences - 2022 - [paper](https://www.mdpi.com/2076-3417/12/11/5614)  [github - not available]()
-  **A RoBERTa Approach for Automated Processing of Sustainability Reports** - Sustainability - 2022 - [paper](https://www.mdpi.com/2071-1050/14/23/16139)  [github - not available]()
-  **Tracking Changes in ESG Representation: Initial Investigations in UK Annual Reports** - Proceedings of CSR-NLP I @LREC 2022 - 2022 - [paper](https://aclanthology.org/2022.csrnlp-1.2/)  [github - not available]()
-  **FinSim4-ESG Shared Task: Learning Semantic Similarities for the Financial Domain. Extended edition to ESG insights** - Proceedings of the Fourth Workshop on Financial Technology and Natural Language Processing (FinNLP@ IJCAIECAI 2022) - 2022 - [paper](https://aclanthology.org/2022.finnlp-1.28/)  [github - not available]()
-  **Natural Language Processing Methods for Scoring Sustainability Reports—A Study of Nordic Listed Companies** - MDPI - Sustainability - 2022 - [paper](https://www.mdpi.com/2071-1050/14/15/9165)  [github - not available]()
-  **Ranking Environment, Social And Governance Related Concepts And Assessing Sustainability Aspect Of Financial Texts** - XXX - 2022 - [paper]()  [github](https://github.com/sohomghosh/Finsim4_ESG)
-  **Do international investors care about ESG news?** - Qualitative Research in Financial Markets - 2022 - [paper](https://www.emerald.com/insight/content/doi/10.1108/QRFM-11-2021-0184/full/html)  [github - not available]()
-  **Predicting Companies’ ESG Ratings from News Articles Using Multivariate Timeseries Analysis** - arxiv - 2022 - [paper](https://arxiv.org/abs/2212.11765)  [github - not available]()
-  **NLP for Responsible Finance: Fine-Tuning Transformer-Based Models for ESG** - 2022 IEEE International Conference on Big Data (Big Data) - 2022 - [paper](https://ieeexplore.ieee.org/abstract/document/10020755)  [github - not available]()
-  **Artificial Intelligence for Sustainable Finance: Why it May Help** - ECMI Research Report - 2022 - [paper](https://www.ceps.eu/wp-content/uploads/2022/12/Artificial-Intelligence-for-Sustainable-Finance.pdf)  [github - not available]()
-  **What do we Learn from a Machine Understanding News Content? Stock Market Reaction to News** - - 2022 - [paper](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4252745)  [github - not available]()
-  **XX** - XXX - 2022 - [paper]()  [github - not available]()
-  **XX** - XXX - 2022 - [paper]()  [github - not available]()
-  **XX** - XXX - 2022 - [paper]()  [github - not available]()



### [2021](#content)

-  **A pretrained language model for climate-related text** - arxiv - 2021 - [paper](https://arxiv.org/abs/2110.12010) [website](https://climatebert.ai/language-model)
-  **Heard the News? Environmental Policy and Clean Investments** - Centre for International Environmental Studies, The Graduate Institute. - 2021 - [paper](https://repository.graduateinstitute.ch/record/299407/)
-  **A NLP-Based Analysis of Alignment of Organizations’ Climate-Related Risk Disclosures with Material Risks and Metrics** - Tackling Climate Change with Machine Learning: workshop at NeurIPS - 2021 - [paper](https://s3.us-east-1.amazonaws.com/climate-change-ai/papers/neurips2021/69/paper.pdf) [github]()
-  **NLP for SDGs: Measuring Corporate Alignment with the Sustainable Development Goals** - Columbia Business School Research Paper - 2021 - [paper](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3874442) [github - not available]()
-  **A Comprehensive Review on Summarizing Financial News Using Deep Learning** - arxiv - 2021 - [paper](https://arxiv.org/abs/2109.10118) [github - not available]()
-  **Automated Identification of Climate Risk Disclosures in Annual Corporate Reports** - arxiv - 2021 - [paper](https://arxiv.org/abs/2108.01415) [github - not available]()
-  **BERT Classification of Paris Agreement Climate Action Plan** - ICML 2021 Workshop on Tackling Climate Change with Machine Learning - 2021 - [paper](https://www.climatechange.ai/papers/icml2021/45) [github - not available]()
-  **XX** - XXX - 2021 - [paper]() [github - not available]()
-  **XX** - XXX - 2021 - [paper]() [github - not available]()



### [2020](#content)


-  **Ask BERT: How Regulatory Disclosure of Transition and Physical Climate Risks affects the CDS Term Structure** - Swiss Finance Institute Research Paper - 2020 - [paper](https://papers.ssrn.com/sol3/Papers.cfm?abstract_id=3616324) [github - not available]()
-  **Analyzing sustainability reports using natural language processing** - arxiv - 2020 - [paper](https://arxiv.org/abs/2011.08073) [github - not available]()
-  **Measuring the readability of sustainability reports: A corpus-based analysis through standard formulae and NLP** - International Journal of Business Communication - 2020 - [paper](https://journals.sagepub.com/doi/pdf/10.1177/2329488416675456) [github - not available]()
-  **ESG2Risk: A Deep Learning Framework from ESG News to Stock Volatility Prediction** - arxiv - 2020 - [paper](https://arxiv.org/abs/2005.02527) [github - not available]()
-  **Mapping ESG Trends by Distant Supervision of Neural Language Models** - mdpi - 2020 - [paper](https://www.mdpi.com/2504-4990/2/4/25) [github - not available]()
-  **XX** - XXX - 2020 - [paper]() [github - not available]()
-  **XX** - XXX - 2020 - [paper]() [github - not available]()
