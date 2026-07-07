import type { Metadata } from "next";
import Link from "next/link";
import { Geist, Geist_Mono } from "next/font/google";
import { DISCLAIMER } from "@/lib/constants";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "İş Kanunu Asistanı",
  description:
    "4857 sayılı İş Kanunu üzerine soru-cevap ve iş sözleşmesi analizi",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html
      lang="tr"
      className={`${geistSans.variable} ${geistMono.variable} h-full antialiased`}
    >
      <body className="min-h-full flex flex-col font-sans">
        <header className="border-b border-gray-200 dark:border-gray-800">
          <nav className="mx-auto flex max-w-3xl items-center gap-6 px-4 py-3">
            <span className="font-semibold">İş Kanunu Asistanı</span>
            <Link href="/" className="text-sm hover:underline">
              Soru-Cevap
            </Link>
            <Link href="/analyze" className="text-sm hover:underline">
              Sözleşme Analizi
            </Link>
          </nav>
        </header>
        <main className="mx-auto w-full max-w-3xl flex-1 px-4 py-6">{children}</main>
        <footer className="border-t border-gray-200 py-3 text-center text-xs text-gray-500 dark:border-gray-800 dark:text-gray-400">
          {DISCLAIMER}
        </footer>
      </body>
    </html>
  );
}
