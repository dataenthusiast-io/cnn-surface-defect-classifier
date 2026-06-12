import type { Metadata } from "next"
import { Geist, Geist_Mono } from "next/font/google"
import "./globals.css"

const geist = Geist({ subsets: ["latin"], variable: "--font-geist-sans" })
const geistMono = Geist_Mono({ subsets: ["latin"], variable: "--font-geist-mono" })

export const metadata: Metadata = {
  title: "CNN Defect Inspector",
  description: "4-Klassen Defektklassifikation · Eigene Produktionsdaten",
}

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="de" className="dark">
      <body className={`${geist.variable} ${geistMono.variable} antialiased bg-zinc-950 text-zinc-100`}>
        {children}
      </body>
    </html>
  )
}
