import numpy as np
from paddleocr import PaddleOCR
import fitz
from PIL import Image
import os
from concurrent.futures import ThreadPoolExecutor
import cv2 as cv

ocr = PaddleOCR(use_angle_cls=True, lang='en') 

def get_paddle_text(img, img_no, file_save=False,output_dir="result"):
    """Use paddleocr to extract text from image"""
    result = ocr.ocr(img, cls=True)

    if not result or not result[0]:
        return ""
    
    txts = '\n'.join([line[1][0] for line in result[0]])
    if file_save:
        os.makedirs(output_dir, exist_ok=True)
        filename = os.path.join(output_dir, f"img_{img_no}.txt")
        with open(filename, "w", encoding="utf-8") as file:
            file.write(txts)

    return txts

def extract_page_image(pdf_document, page_number):
    """Extracts an image from a single PDF page."""
    page = pdf_document.load_page(page_number)
    pix = page.get_pixmap()
    image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    return image

def get_image_from_pdf(pdf_path):
    """Extract images from a PDF using multi-threading."""
    pdf_document = fitz.open(pdf_path)
    total_pages = len(pdf_document)

    pil_image_list = []
    with ThreadPoolExecutor() as executor:
        pil_image_list = list(executor.map(lambda p: extract_page_image(pdf_document, p), range(total_pages)))

    # Convert PIL images to OpenCV format
    cv_images_list = [cv.cvtColor(np.array(img), cv.COLOR_RGB2BGR) for img in pil_image_list]

    return pil_image_list, cv_images_list

def get_all_ocr_text(img_list,output_dir):
    """Extract OCR text from images using multithreading."""
    ocr_result_list = []
    for i,j in enumerate(img_list):
        res = get_paddle_text(img = j,
                       file_save=True,
                        img_no=i,
                        output_dir=output_dir)
        ocr_result_list.append(res)
    
    return ocr_result_list

def start_process(pdf_path,output_dir):
    """Start pdf processing."""
    pil_images, cv_images = get_image_from_pdf(pdf_path)
    res = get_all_ocr_text(cv_images,output_dir)

if __name__ == '__main__':
    current_directory = os.getcwd()
    file_name = "Gov_land_records.pdf"
    output_dir = "result"
    full_path = os.path.join(current_directory, file_name)
    start_process(full_path)
