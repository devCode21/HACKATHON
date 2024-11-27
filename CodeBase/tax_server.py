from http.server import BaseHTTPRequestHandler, HTTPServer
import json

class TaxRequestHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path == "/optimize_tax":  # Correct path for the request
            try:
                # Get the content length (size of incoming data)
                content_length = int(self.headers["Content-Length"])
                # Read and parse the incoming data
                post_data = self.rfile.read(content_length)
                data = json.loads(post_data)

                # Extract data from the POST request
                annual_income = float(data.get("annual_income", 0))
                basic_salary = float(data.get("basic_salary", 0))
                hra_rent_paid = float(data.get("hra_rent_paid", 0))
                health_insurance_premium = float(data.get("health_insurance_premium", 0))
                education_loan_interest = float(data.get("education_loan_interest", 0))
                investments_80c = float(data.get("investments_80c", 0))
                home_loan_interest = float(data.get("home_loan_interest", 0))

                # Tax calculation logic
                total_deductions = (
                    hra_rent_paid
                    + health_insurance_premium
                    + education_loan_interest
                    + investments_80c
                    + home_loan_interest
                )
                taxable_income = annual_income - total_deductions
                estimated_tax = taxable_income * 0.1 if taxable_income > 0 else 0

                # Prepare the response data
                response = {
                    "annual_income": annual_income,
                    "total_deductions": total_deductions,
                    "taxable_income": taxable_income,
                    "estimated_tax": estimated_tax,
                    "deduction_details": {
                        "HRA Rent Paid": hra_rent_paid,
                        "Health Insurance Premium": health_insurance_premium,
                        "Education Loan Interest": education_loan_interest,
                        "80C Investments": investments_80c,
                        "Home Loan Interest": home_loan_interest,
                    },
                }

                # Debug: print the response to check if deduction_details are there
                print("Response data: ", response)

                # Send the response back to the client
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")  # Add CORS header
                self.end_headers()
                self.wfile.write(json.dumps(response).encode("utf-8"))
            except Exception as e:
                # Handle errors and send an error response
                print("Error in processing POST request:", e)  # Debugging error
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                error_response = {"error": str(e)}
                self.wfile.write(json.dumps(error_response).encode("utf-8"))

    # Handle CORS preflight request
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()


# Start the server
def run(server_class=HTTPServer, handler_class=TaxRequestHandler, port=8000):
    server_address = ("", port)
    httpd = server_class(server_address, handler_class)
    print(f"Starting server on port {port}...")
    httpd.serve_forever()

if __name__ == "__main__":
    run()
