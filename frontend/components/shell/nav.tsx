"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const LINKS = [
  { href: "/", label: "Overview" },
  { href: "/analyze", label: "Audio Analysis" },
  { href: "/live", label: "Voice Control" },
  { href: "/dashboard", label: "Evaluation" },
];

export function Nav() {
  const pathname = usePathname();
  return (
    <nav className="nav" aria-label="Main">
      {LINKS.map((link) => (
        <Link
          key={link.href}
          href={link.href}
          aria-current={
            (link.href === "/" ? pathname === "/" : pathname.startsWith(link.href)) ? "page" : undefined
          }
        >
          {link.label}
        </Link>
      ))}
    </nav>
  );
}
