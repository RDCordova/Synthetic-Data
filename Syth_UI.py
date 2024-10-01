import streamlit as st
import boto3
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from botocore.config import Config
from langchain_community.chat_models import BedrockChat
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import os
import time 
import zipfile

# Reuse your existing functions from the provided code

# Step 1: Start Textract Asynchronous Job for Large PDFs or Complex Documents
def start_textract_job(bucket_name, document_name):
    textract = boto3.client('textract')

    response = textract.start_document_analysis(
        DocumentLocation={'S3Object': {'Bucket': bucket_name, 'Name': document_name}},
        FeatureTypes=['TABLES', 'FORMS']
    )
    return response['JobId']


def check_textract_job_status(job_id):
    textract = boto3.client('textract')
    while True:
        response = textract.get_document_analysis(JobId=job_id)
        status = response['JobStatus']
        if status == 'SUCCEEDED':
            return response
        elif status == 'FAILED':
            raise Exception(f"Textract job failed: {response}")
        time.sleep(5)


def extract_text_and_tables_with_layout_from_s3_async(bucket_name, document_name):
    job_id = start_textract_job(bucket_name, document_name)
    response = check_textract_job_status(job_id)

    blocks = response['Blocks']
    text_blocks = []
    table_blocks = []

    def extract_table_data_with_layout(relationship_block, block_map):
        table_data = []
        for relationship in relationship_block['Relationships']:
            if relationship['Type'] == 'CHILD':
                for child_id in relationship['Ids']:
                    cell_block = block_map[child_id]
                    if cell_block['BlockType'] == 'CELL':
                        cell_text = cell_block.get('Text', "")
                        bbox = cell_block['Geometry']['BoundingBox']
                        table_data.append({
                            'text': cell_text,
                            'bbox': bbox
                        })
        return table_data

    block_map = {block['Id']: block for block in blocks}

    for block in blocks:
        if block['BlockType'] == 'LINE':
            bbox = block['Geometry']['BoundingBox']
            text_blocks.append({'text': block['Text'], 'bbox': bbox})
        elif block['BlockType'] == 'TABLE':
            table_data = extract_table_data_with_layout(block, block_map)
            table_blocks.append(table_data)

    return text_blocks, table_blocks


def load_model(top_k, top_p, temperature):
    config = Config(read_timeout=1000)
    bedrock_runtime = boto3.client(service_name='bedrock-runtime', region_name='us-east-1', config=config)

    model_id = "anthropic.claude-3-5-sonnet-20240620-v1:0"
    
    model_kwargs = { 
        "max_tokens": 100000,
        "temperature": temperature,
        "top_k": top_k,
        "top_p": top_p,
        "stop_sequences": ["\n\nHuman"],
    }
    
    model = BedrockChat(client=bedrock_runtime, model_id=model_id, model_kwargs=model_kwargs)
    return model


def generate_new_content_with_mapping(text_blocks, model):
    new_text_blocks = []

    def process_text_block(block):
        prompt = f"Rewrite the following content. Only return the new version, do not include any sort of explanations or prefixes. For example do not include phrases like Here is a new version of the content, preserving the original format :\n\n{block['text']}"
        response = model.invoke(prompt).content
        return {'text': response, 'bbox': block['bbox']}

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(process_text_block, block) for block in text_blocks]

        for future in tqdm(as_completed(futures), total=len(futures), desc="Generating new content", unit="block"):
            new_text_blocks.append(future.result())

    return new_text_blocks


def create_pdf_with_mapped_content(mapped_text_blocks, tables, output_file):
    c = canvas.Canvas(output_file, pagesize=letter)
    width, height = letter
    current_y = height - 50

    def scale_bbox_to_page(bbox, width, height):
        x = bbox['Left'] * width
        y = height - (bbox['Top'] * height)
        w = bbox['Width'] * width
        h = bbox['Height'] * height
        return x, y, w, h

    for block in mapped_text_blocks:
        text = block['text']
        bbox = block['bbox']
        x, y, w, h = scale_bbox_to_page(bbox, width, height)

        if current_y - h < 50:
            c.showPage()
            current_y = height - 50

        c.setFont("Helvetica", 10)
        c.drawString(x, current_y, text)
        current_y -= h + 10

    for table in tables:
        table_height = sum([cell['bbox']['Height'] * height for cell in table])

        if current_y - table_height < 50:
            c.showPage()
            current_y = height - 50

        for cell in table:
            cell_text = cell['text']
            bbox = cell['bbox']
            x, y, w, h = scale_bbox_to_page(bbox, width, height)

            if current_y - h < 50:
                c.showPage()
                current_y = height - 50

            c.setFont("Helvetica", 10)
            c.drawString(x, current_y, cell_text)
            c.rect(x, current_y - h, w, h)
            current_y -= h + 10

    c.save()

def create_zip_from_pdfs(pdf_files, output_zip_file):
    with zipfile.ZipFile(output_zip_file, 'w') as zipf:
        for pdf_file in pdf_files:
            zipf.write(pdf_file, os.path.basename(pdf_file))
            
def process_pdf_from_s3_with_mapped_content(bucket_name, document_name, output_pdf, top_k, top_p, temperature):
    extracted_text_blocks, table_blocks = extract_text_and_tables_with_layout_from_s3_async(bucket_name, document_name)
    llm = load_model(top_k, top_p, temperature)
    mapped_text_blocks = generate_new_content_with_mapping(extracted_text_blocks, llm)
    create_pdf_with_mapped_content(mapped_text_blocks, table_blocks, output_pdf)

def get_s3_client():
    return boto3.client('s3',region='us-east-1')

# Streamlit UI
st.set_page_config(
    page_title="SyntheSys: Content Generation",
    page_icon="🧊",
    layout="centered")
st.title("SyntheSys: Content Generation")
st.write("Select the data type to generate (PDF or Tabular) then upload your sample file")

st.markdown("---")

# Step 1: User selects between PDF or Tabular Data Generation
generation_type = st.radio("Select Generation Type", ["Generate PDF(s)", "Generate Tabular Data"])


def get_s3():
    return boto3.client('s3')

def upload_file_to_s3(file, bucket_name, object_name):
    s3 = get_s3()
    try: 
        s3.upload_fileobj(file, bucket_name, object_name)
    except NoCredentialsError:
        st.error("CA")
    except ClientError:
        st.error("Fail")
        
# If user selects PDF Generation
if generation_type == "Generate PDF(s)":
    st.write("### PDF Generation")
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
    # Sidebar 
    st.sidebar.title("Generation Options")
    # st.sidebar.header("Select number of PDF to genrate")
    num_pdfs = st.sidebar.slider("Number of PDFs to Generate", min_value=1, max_value=10, value=1)
    st.sidebar.header("Adjust PDF Generation Parameters")
    top_k = st.sidebar.slider("Top K (Token Sampling)", min_value=1, max_value=1000, value=250, step=1)
    top_p = st.sidebar.slider("Top P (Probability Threshold)", min_value=0.0, max_value=1.0, value=0.9, step=0.01)
    temperature = st.sidebar.slider("Temperature (Creativity Control)", min_value=0.0, max_value=1.0, value=0.0, step=0.01)

    if uploaded_file is not None:
        bucket_name = 'a31-cd'
        object_name = uploaded_file.name
        upload_file_to_s3(uploaded_file, bucket_name, object_name)
        st.success("PDF uploaded successfully!")
        
        

        if st.button("Generate PDFs"):
            with st.spinner("Generating Data...PDFs..."):
                bucket_name = 'a31-cd'  # Set your S3 bucket name

                def generate_pdfs(num_pdfs):
                    output_files = []
                    for i in range(num_pdfs):
                        output_pdf = f"generated_pdf_{i+1}.pdf"
                        process_pdf_from_s3_with_mapped_content(bucket_name,
                                                                object_name,
                                                                output_pdf,
                                                                top_k,
                                                                top_p,
                                                                temperature)
                        output_files.append(output_pdf)
                    return output_files

                generated_pdfs = generate_pdfs(num_pdfs)
                
                if num_pdfs > 1:
                    zip_name = "generated_pdfs.zip"
                    create_zip_from_pdfs(generated_pdfs,zip_name)
                    
                    with open(zip_name, "rb") as file:
                        st.download_button(f"Download All PDFs as {zip_name}", file, zip_name, mime="application/zip")
                else:
                    pdf_file = generated_pdfs[0]
                    st.success(f"PDF generated successfully")
                    with open(pdf_file, "rb") as file:
                        st.download_button(f"Download PDF", file, pdf_file, mime= "application/zip")
                

                # for i, pdf_file in enumerate(generated_pdfs):
                #     st.success(f"PDF {i+1} generated successfully!")
                #     with open(pdf_file, "rb") as file:
                #         st.download_button(f"Download PDF {i+1}", file, f"generated_pdf_{i+1}.pdf", mime="application/pdf")

# If user selects Tabular Data Generation
else:
    st.write("### Tabular Data Generation")
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    # Sidebar 
    st.sidebar.title("Generation Options")
    # st.sidebar.header("Select number of PDF to genrate")
    rows = st.sidebar.slider("Number of obervations", min_value=10, max_value=1000, value=100, step=10)
    st.sidebar.header("Adjust Data Generation Parameters")
    top_k = st.sidebar.slider("Top K (Token Sampling)", min_value=1, max_value=1000, value=250, step=1)
    top_p = st.sidebar.slider("Top P (Probability Threshold)", min_value=0.0, max_value=1.0, value=0.9, step=0.01)
    temperature = st.sidebar.slider("Temperature (Creativity Control)", min_value=0.0, max_value=1.0, value=0.0, step=0.01)

    from io import StringIO
    import pandas as pd
    
    def read_csv_from_s3(bucket_name, key):
        s3 =  boto3.client('s3') 
        # objects = s3.list_objects_v2(Bucket=bucket_name)
        # if objects.get('Contents'):
        #     st.info(f"Files in bucket {bucket_name}: {[obj['Key'] for obj in objects['Contents']]}")
        obj = s3.get_object(Bucket=bucket_name,Key=key)
        df = pd.read_csv(StringIO(obj['Body'].read().decode('utf-8')))
        # st.success("Yay")
        return df
    if uploaded_file is not None:
        bucket_name = 'a31-cd'
        object_name = uploaded_file.name
        upload_file_to_s3(uploaded_file, bucket_name, object_name)
        st.success("PDF uploaded successfully!")
        
        if st.button("Generate Data"):
            df = read_csv_from_s3(bucket_name,'tips.csv')
            sub = df.sample(10)

            def convert_do_to_csv(df):
                return df.to_csv(index=False).encode('utf-8')

            csv_data = convert_do_to_csv(sub)

            st.download_button(f"Download Data", csv_data, 'generated_data.csv', mime= "text/csv")
            
        
        


        
        
                     
                                              
    
                                            

    
    # if st.button("Generate Data"):
    #     with st.spinner("Generating Data..."):