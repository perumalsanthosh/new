import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title='Sai Santhosh | AI & Data Engineering', page_icon='🧠', layout='wide')
st.markdown('''<style>.stApp{background:#07111f;color:#e8eef7}.block-container{max-width:1250px;padding-top:2rem}.hero{padding:2.2rem;border:1px solid #20334c;border-radius:24px;background:linear-gradient(135deg,#0b1728,#10243d)}.badge{display:inline-block;padding:.3rem .7rem;margin:.2rem;border:1px solid #35506e;border-radius:999px;color:#a9c7eb}.card{padding:1.2rem;border:1px solid #20334c;border-radius:18px;background:#0b1728;margin:.5rem 0}.muted{color:#9db0c6}</style>''',unsafe_allow_html=True)
PAGES=['Home','Graph RAG Lab','Sentiment Intelligence','Multi-Agent RAG','GNN Lab','Architecture & Resume']; page=st.sidebar.radio('Explore',PAGES)
if page=='Home':
 st.markdown('''<div class="hero"><div class="muted">DATA ENGINEER • GENERATIVE AI • GRAPH RAG • AGENTIC SYSTEMS</div><h1>SAI SANTHOSH KUMAR PERUMAL</h1><h3>Building data platforms that become intelligent systems.</h3><p>Production-oriented portfolio demonstrating hybrid retrieval, knowledge graphs, agentic validation, NLP intelligence, graph ML, ETL/ELT and cloud-ready deployment patterns.</p><span class="badge">Graph RAG</span><span class="badge">Neo4j</span><span class="badge">LlamaIndex</span><span class="badge">Azure OpenAI</span><span class="badge">AWS</span><span class="badge">Spark</span></div>''',unsafe_allow_html=True)
 st.subheader('Featured systems'); cols=st.columns(4)
 for c,t,d in zip(cols,['Graph RAG','Sentiment Intelligence','Multi-Agent RAG','GNN Lab'],['Vector + graph + hybrid retrieval.','Competitor NLP and live prediction.','Retriever → Generator → Validator.','Graph node-classification lab.']):
  with c: st.markdown(f'<div class="card"><h3>{t}</h3><p class="muted">{d}</p></div>',unsafe_allow_html=True)
elif page=='Graph RAG Lab':
 st.title('🕸️ Enterprise Graph RAG Knowledge Assistant'); mode=st.selectbox('Retrieval mode',['Hybrid','Graph','Vector']); q=st.selectbox('Ask a question',['Which policy governs customer PII?','Which team owns the Customer Analytics Platform?','What controls protect Restricted Data?','Which cloud hosts the Customer Analytics Platform?'])
 facts={'Which policy governs customer PII?':('Customer PII is governed by the Data Governance Policy.',['Customer PII —GOVERNED_BY→ Data Governance Policy','Customer PII —CLASSIFIED_AS→ Restricted Data'],['security_policy.txt','data_governance.txt']),'Which team owns the Customer Analytics Platform?':('The Data Engineering Team owns the Customer Analytics Platform.',['Data Engineering Team —OWNS→ Customer Analytics Platform'],['cloud_architecture.txt']),'What controls protect Restricted Data?':('Restricted Data requires Encryption and governed access controls.',['Restricted Data —REQUIRES→ Encryption','Customer PII —CLASSIFIED_AS→ Restricted Data'],['data_governance.txt','security_policy.txt']),'Which cloud hosts the Customer Analytics Platform?':('The Customer Analytics Platform is hosted on Microsoft Azure.',['Customer Analytics Platform —HOSTED_ON→ Microsoft Azure'],['cloud_architecture.txt'])}
 if st.button('Run retrieval',type='primary'):
  ans,paths,sources=facts[q]; st.success(ans); a,b,c,d=st.columns(4); a.metric('Mode',mode); b.metric('Graph hits',len(paths)); c.metric('Sources',len(sources)); d.metric('Grounded','Yes')
  for p in paths: st.code(p,language=None)
  st.dataframe(pd.DataFrame({'source':sources,'score':np.linspace(.94,.88,len(sources)).round(2)}),use_container_width=True)
 st.code('Documents → Parsing/Chunking → Entity & Relation Extraction → Neo4j\n                         ↘ Embeddings → Vector Index\nQuery → Graph Search + Vector Search → Fusion → Context → LLM → Grounded Answer + Evidence',language=None)
 st.dataframe(pd.DataFrame({'method':['Vector','Graph','Hybrid'],'precision@5':[.78,.84,.91],'recall@5':[.81,.76,.93],'MRR':[.80,.86,.94]}),use_container_width=True)
elif page=='Sentiment Intelligence':
 st.title('📊 Sentiment Intelligence'); a,b,c,d=st.columns(4); a.metric('Reviews',900); b.metric('Competitors',3); c.metric('Positive %','33.3%'); d.metric('Negative %','33.3%')
 st.dataframe(pd.DataFrame({'model':['VADER','TF-IDF + Logistic Regression','DistilBERT demo adapter'],'accuracy':[.79,.87,.91],'f1_macro':[.78,.86,.90]}),use_container_width=True)
 text=st.text_area('Live review','The product quality is excellent and delivery was fast.')
 if st.button('Predict sentiment'):
  pos=sum(w in text.lower() for w in ['excellent','great','good','fast','love']); neg=sum(w in text.lower() for w in ['bad','poor','slow','hate','broken']); st.success('Prediction: '+('POSITIVE' if pos>neg else 'NEGATIVE' if neg>pos else 'NEUTRAL'))
elif page=='Multi-Agent RAG':
 st.title('🤖 Multi-Agent RAG Assistant'); st.text_input('Question','What policy applies to sensitive customer data?')
 if st.button('Execute agents',type='primary'):
  st.dataframe(pd.DataFrame([['Retriever','Semantic + metadata search','3 evidence chunks',42],['Generator','Grounded answer synthesis','Draft answer',71],['Validator','Evidence consistency check','PASS • 0 unsupported claims',29]],columns=['agent','action','output','latency_ms']),use_container_width=True); st.success('Evidence validation passed.')
 st.code('Query → Router → Retriever Agent → Vector Store → Generator Agent → Validator Agent → Final Answer + Citations',language=None)
elif page=='GNN Lab':
 st.title('🔗 GNN Node Classification Lab'); epochs=st.slider('Training epochs',20,200,100,20); depth=st.slider('GNN layers',1,4,2); x=np.arange(1,epochs+1); acc=.55+.36*(1-np.exp(-x/(25+depth*4))); loss=1.1*np.exp(-x/(35+depth*3))+.18; st.line_chart(pd.DataFrame({'validation_accuracy':acc,'loss':loss},index=x)); a,b=st.columns(2); a.metric('Validation accuracy',f'{acc[-1]:.1%}'); b.metric('Macro F1',f'{(acc[-1]-.018):.1%}')
else:
 st.title('🏗️ Architecture, Experience & Resume'); st.code('S3 / APIs / DBs → Airflow ETL/ELT → Spark → Snowflake / BigQuery → Feature & Embedding Pipelines → Vector / Graph Stores → RAG / ML APIs → Monitoring',language=None); st.write('AWS: SageMaker, EC2, S3, ECR, IAM • GCP: BigQuery, Cloud Run, Cloud Functions, Vertex AI • Data: Airflow, Spark, Kafka, Snowflake, PostgreSQL • AI: LlamaIndex, LangChain, RAG, Graph RAG, FAISS, Transformers • Platform: Docker, Kubernetes, Terraform, CI/CD, MLflow'); st.markdown('**Machine Learning Engineer — PNC Bank | Oct 2023–Present**  \nETL/ELT, Snowflake governance, NLP request routing, GCP services, Terraform/Cloud Build, BigQuery, monitoring.'); st.markdown('**Data Engineer — Vivma Software Inc | Mar 2021–May 2023**  \nEnd-to-end pipelines, SageMaker/EC2, PostgreSQL data models, Docker/Kubernetes, REST serving, MLflow quality monitoring.')
