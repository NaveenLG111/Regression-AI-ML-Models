import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Walmart Weekly Sales Forecaster",
  description: "End-to-End Machine Learning Sales Forecast",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="bg-slate-900 text-slate-100">{children}</body>
    </html>
  );
}
