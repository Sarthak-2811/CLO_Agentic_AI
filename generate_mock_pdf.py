# generate_mock_pdf.py
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import os

def create_mock_indenture(filename="mock_indenture.pdf"):
    # Ensure data directory exists
    os.makedirs("data/raw_indentures", exist_ok=True)
    filepath = os.path.join("data/raw_indentures", filename)
    
    c = canvas.Canvas(filepath, pagesize=letter)
    
    # Page 1: Tranches & Fees
    c.setFont("Helvetica-Bold", 16)
    c.drawString(72, 700, "INDENTURE: Octagon Investment Partners Mock CLO")
    
    c.setFont("Helvetica", 12)
    text_page_1 = [
        "1. Capital Structure:",
        "The target par amount of the collateral obligations is $400,000,000.",
        "Class A-1 Notes: $250,000,000. Target Rating: AAA. Interest Rate: SOFR + 1.35%.",
        "Class B Notes: $40,000,000. Target Rating: AA. Interest Rate: SOFR + 2.00%.",
        "Class C Notes: $30,000,000. Target Rating: BBB. Interest Rate: SOFR + 3.10%.",
        "Subordinated Notes (Equity): $80,000,000. Target Rating: NR. Interest Rate: Residual.",
        "",
        "2. Administrative and Management Fees:",
        "The Senior Administrative Fee Cap is $200,000 per annum.",
        "The Senior Management Fee is 0.15% (15 bps) of the collateral balance.",
        "The Subordinated Management Fee is 0.35% (35 bps) of the collateral balance.",
        "The Incentive Fee Hurdle is a 12.0% IRR to the Subordinated Notes.",
        "The Manager receives 20% of residual cash flows above the Incentive Fee Hurdle."
    ]
    
    y = 660
    for line in text_page_1:
        c.drawString(72, y, line)
        y -= 20
        
    c.showPage()
    
    # Page 2: Covenants & Waterfall
    c.setFont("Helvetica-Bold", 16)
    c.drawString(72, 700, "ARTICLE 11: PRIORITY OF PAYMENTS & COVENANTS")
    
    c.setFont("Helvetica", 12)
    text_page_2 = [
        "3. Coverage Tests:",
        "Class A/B Overcollateralization (OC) Trigger: 121.5%.",
        "Class C Overcollateralization (OC) Trigger: 114.0%.",
        "Class A/B Interest Coverage (IC) Trigger: 120.0%.",
        "If any Coverage Test is breached, Interest Proceeds shall be diverted to pay down",
        "Senior principal until the test is satisfied.",
        "",
        "4. Portfolio Constraints:",
        "The Maximum CCC-rated obligation limit is 7.5% of the total portfolio.",
        "",
        "5. Interest Proceeds Waterfall (Priority of Payments):",
        "1. Taxes and Trustee Fees (subject to cap).",
        "2. Senior Management Fee.",
        "3. Class A-1 Interest.",
        "4. Class B Interest.",
        "5. Class A/B Coverage Tests (Cure if breached).",
        "6. Class C Interest.",
        "7. Class C Coverage Test (Cure if breached).",
        "8. Subordinated Management Fee.",
        "9. Residual to Subordinated Notes."
    ]
    
    y = 660
    for line in text_page_2:
        c.drawString(72, y, line)
        y -= 20
        
    c.save()
    print(f"Successfully generated {filepath}")

if __name__ == "__main__":
    create_mock_indenture()