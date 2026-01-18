import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'Room Identity Optimizer',
  description: 'Transform your room to support who you want to be',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="antialiased">{children}</body>
    </html>
  );
}


