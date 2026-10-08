import React from "react";
import "./globals.css";

export const metadata = {
  title: "Cancer Mortality ML Platform",
  description: "Leak-Proof Regularized Gradient Boosting Pipeline",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <head>
        <script src="https://cdn.tailwindcss.com"></script>
      </head>
      <body className="bg-slate-950 text-slate-100 antialiased">
        {children}
      </body>
    </html>
  );
}
