import os
import PyPDF2
import fitz
import torch
from transformers import (
    AutoModelForQuestionAnswering, 
    AutoTokenizer, 
    pipeline
)
import numpy as np

class PDFChatbot:
    def __init__(self, model_name='deepset/roberta-base-squad2'):
        """
        Initialize the chatbot with a pre-trained question-answering model
        
        Args:
            model_name (str): Hugging Face model for question answering
        """
        # Check for GPU availability
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # Load tokenizer and model
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForQuestionAnswering.from_pretrained(model_name).to(self.device)
            
            # Create question-answering pipeline
            self.qa_pipeline = pipeline(
                "question-answering", 
                model=self.model, 
                tokenizer=self.tokenizer,
                device=0 if self.device == "cuda" else -1
            )
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
        
        # PDF text storage
        self.pdf_text = ""
        self.chunks = []
    
    def load_pdf(self, pdf_path, chunk_size=2000, overlap=500):
        """
        Advanced PDF loading with larger chunk sizes using PyMuPDF
        
        Args:
            pdf_path (str): Path to the PDF file
            chunk_size (int): Size of text chunks to create
            overlap (int): Number of characters to overlap between chunks
        
        Returns:
            bool: True if PDF loaded successfully, False otherwise
        """
        try:
            # Open the PDF document
            doc = fitz.open(pdf_path)
            
            # Extract text from all pages
            self.pdf_text = ""
            for page in doc:
                self.pdf_text += page.get_text()
            
            # Improved chunking strategy
            self.chunks = []
            for i in range(0, len(self.pdf_text), chunk_size - overlap):
                chunk = self.pdf_text[i:i+chunk_size]
                self.chunks.append(chunk)
            
            # Print loading information
            print(f"PDF loaded. Total text length: {len(self.pdf_text)} characters")
            print(f"Split into {len(self.chunks)} chunks")
            
            # Close the document
            doc.close()
            
            return True
        
        except Exception as e:
            print(f"Error loading PDF: {e}")
            return False
    
    def answer_question(self, question, top_k=3):
        """
        Answer question using multiple chunks and aggregate results
        
        Args:
            question (str): User's question
            top_k (int): Number of top answers to return
        
        Returns:
            list: Top answers with confidence scores
        """
        if not self.chunks:
            return [{"answer": "No PDF loaded. Please load a PDF first.", "score": 0}]
        
        # Perform QA on each chunk
        results = []
        for chunk in self.chunks:
            try:
                # Run question-answering pipeline
                result = self.qa_pipeline({
                    'question': question,
                    'context': chunk
                })
                
                # Only add if answer is not empty and meets minimum confidence
                if result['answer'] and result['score'] > 0.1:
                    results.append(result)
            except Exception as e:
                print(f"Error processing chunk: {e}")
        
        # Sort results by confidence score
        results.sort(key=lambda x: x['score'], reverse=True)
        
        # Return top k results
        return results[:top_k] if results else [
            {"answer": "Sorry, I didn't understand your question. Do you want to connect with a live agent?", "score": 0}
        ]

def main():
    # Initialize chatbot
    chatbot = PDFChatbot()
    
    # Welcome message
    print("=== PDF Chatbot with Hugging Face ===")
    print("Commands:")
    print("  /load <path_to_pdf> - Load a PDF file")
    print("  /quit - Exit the application")
    print("  Type your question after loading a PDF")
    
    # Main interaction loop
    while True:
        # Get user input
        user_input = input("\n> ").strip()
        
        # Check for special commands
        if user_input.lower() == '/quit':
            print("Goodbye!")
            break
        
        # Load PDF command
        if user_input.startswith('/load'):
            try:
                pdf_path = user_input.split(' ', 1)[1]
                print(f"Attempting to load PDF: {pdf_path}")
                
                if os.path.exists(pdf_path):
                    if chatbot.load_pdf(pdf_path):
                        print(f"PDF loaded successfully: {pdf_path}")
                    else:
                        print("Failed to load PDF")
                else:
                    print(f"File not found: {pdf_path}")
            except IndexError:
                print("Please provide a PDF file path. Usage: /load /path/to/your/document.pdf")
            continue
        
        # Question answering
        if chatbot.chunks:
            # Get answers
            answers = chatbot.answer_question(user_input)
            
            # Display results
            print("\n[Answers]:")
            for i, result in enumerate(answers, 1):
                print(f"{i}. Confidence: {result['score']:.2f}")
                print(f"   {result['answer']}\n")
        else:
            print("\n[System] Please load a PDF first using /load command")

if __name__ == "__main__":
    main()
    