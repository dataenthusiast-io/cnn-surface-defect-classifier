import { cn } from "@/lib/utils"

interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: "default" | "outline" | "ghost" | "destructive"
  size?: "sm" | "md" | "lg"
}

export function Button({ className, variant = "default", size = "md", ...props }: ButtonProps) {
  return (
    <button
      className={cn(
        "inline-flex items-center justify-center gap-1.5 rounded-lg font-medium font-mono text-xs transition-colors focus-visible:outline-none disabled:pointer-events-none disabled:opacity-40",
        variant === "default"     && "bg-zinc-800 text-zinc-100 hover:bg-zinc-700 border border-zinc-700",
        variant === "outline"     && "border border-zinc-700 bg-transparent text-zinc-300 hover:bg-zinc-800",
        variant === "ghost"       && "text-zinc-400 hover:bg-zinc-800 hover:text-zinc-200",
        variant === "destructive" && "bg-red-900/40 text-red-400 border border-red-800 hover:bg-red-900/60",
        size === "sm" && "h-8 px-3",
        size === "md" && "h-9 px-4",
        size === "lg" && "h-11 px-6",
        className,
      )}
      {...props}
    />
  )
}
