import type { Metadata } from "next";
import Link from "next/link";
import type { ReactNode } from "react";
import { BrandMark } from "../components/icons";
import { BackendPill, BackendProvider } from "../components/shell/backend";
import { Nav } from "../components/shell/nav";
import "./globals.css";

export const metadata: Metadata = {
  title: "KWS Lab · Keyword Spotting",
  description: "Local LSTM keyword spotting: audio analysis, voice control, and run-based evaluation.",
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body>
        <BackendProvider>
          <header className="shell-header">
            <div className="shell-header-inner">
              <Link href="/" className="brand">
                <BrandMark />
                KWS Lab
              </Link>
              <Nav />
              <BackendPill />
            </div>
          </header>
          {children}
        </BackendProvider>
      </body>
    </html>
  );
}
