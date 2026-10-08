import React from "react";

export const metadata = {
  title: "Retail Price Optimization Engine",
  description: "Ridge Regression Pricing Dashboard",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
