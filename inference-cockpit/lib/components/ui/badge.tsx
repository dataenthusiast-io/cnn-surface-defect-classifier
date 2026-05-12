import { cn } from "@/lib/utils"

interface BadgeProps extends React.HTMLAttributes<HTMLSpanElement> {
  variant?: "default" | "destructive" | "success" | "outline"
}

export function Badge({ className, variant = "default", ...props }: BadgeProps) {
  return (
    <span
      className={cn(
        "inline-flex items-center rounded-full px-3 py-1 text-sm font-semibold transition-colors",
        variant === "default"     && "bg-slate-100 text-slate-800",
        variant === "destructive" && "bg-red-600 text-white",
        variant === "success"     && "bg-green-600 text-white",
        variant === "outline"     && "border border-slate-300 text-slate-700",
        className
      )}
      {...props}
    />
  )
}
