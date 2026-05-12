"use client"
import dynamic from "next/dynamic"

const CockpitDashboard = dynamic(
  () => import("@/lib/components/cockpit-dashboard").then(m => m.CockpitDashboard),
  { ssr: false }
)

export default function Home() {
  return <CockpitDashboard />
}
