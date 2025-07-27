import React from "react";
import "./Pricing.css";

const plans = [
  {
    model: "Freemium",
    segment: "Individual students and teachers",
    price: "Free",
    notes: "Limited features, suitable for basic use",
  },
  {
    model: "Paid Monthly Plan",
    segment: "Individual students and teachers",
    price: "199/month",
    notes: "Access to all features, billed monthly",
  },
  {
    model: "Paid Annual Plan",
    segment: "Individual students and teachers",
    price: "1999/year",
    notes: "Access to all features, billed annually (save 16%)",
  },
  {
    model: "Bulk Licenses",
    segment: "Schools and institutions",
    price: "Custom pricing",
    notes: "Contact us for a quote based on your needs",
  },
  {
    model: "Hardware Bundle",
    segment: "Schools and institutions",
    price: "Custom pricing",
    notes: "Includes tablets and other hardware, contact us for a quote",
  },
  {
    model: "SMS/USSD Packs",
    segment: "All users",
    price: "Starting from 99",
    notes: "Purchase packs for SMS and USSD functionality",
  },
];

export default function Pricing() {
  return (
    <div className="container">
      <div className="header">Pricing</div>
      <div className="subheader">
        Choose the plan that fits your needs. Start with a free trial and upgrade as you go.
      </div>
      <div className="tableWrapper">
        <table className="table">
          <thead>
            <tr>
              <th>Model</th>
              <th>Target Segment</th>
              <th>Price Point (INR)</th>
              <th>Notes</th>
            </tr>
          </thead>
          <tbody>
            {plans.map((plan, i) => (
              <tr key={i}>
                <td>{plan.model}</td>
                <td>
                  {(plan.segment === "Individual students and teachers" ||
                    plan.segment === "Schools and institutions") ? (
                    <a href="/">{plan.segment}</a>
                  ) : (
                    plan.segment
                  )}
                </td>
                <td>{plan.price}</td>
                <td>{plan.notes}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="footer">
        All prices are in Indian Rupees (INR) and exclude applicable taxes.
        Contact us for enterprise solutions and custom pricing.
      </div>
    </div>
  );
}