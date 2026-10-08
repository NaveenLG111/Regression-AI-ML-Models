import "./globals.css";

export const metadata = {
  title: "Paddy Prediction ML",
  description: "ML Model Dashboard",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="bg-slate-50">{children}</body>
    </html>
  );
}
