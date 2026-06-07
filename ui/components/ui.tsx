import clsx from "clsx";
import { KeyboardEvent, ReactNode } from "react";

export function Card({
  title,
  children,
  className,
  badge,
}: {
  title: string;
  children: ReactNode;
  className?: string;
  badge?: ReactNode;
}) {
  return (
    <div
      className={clsx(
        "animate-fade-in rounded-xl border border-line bg-panel/80 p-5 shadow-card backdrop-blur-sm transition-colors duration-200 hover:border-line-strong",
        className,
      )}
    >
      <h2 className="mb-4 flex items-center gap-2 text-[12px] font-semibold uppercase tracking-[0.14em] text-muted">
        <span className="h-3 w-1 rounded-full bg-accent-grad" />
        {title}
        {badge && <span className="ml-1 inline-block normal-case">{badge}</span>}
      </h2>
      {children}
    </div>
  );
}

export function Pill({
  children,
  variant = "muted",
}: {
  children: ReactNode;
  variant?: "good" | "bad" | "warn" | "muted";
}) {
  const styles = {
    good: "border-good/30 bg-good/10 text-good",
    bad: "border-bad/30 bg-bad/10 text-bad",
    warn: "border-warn/30 bg-warn/10 text-warn",
    muted: "border-line bg-panel2 text-muted",
  };
  return (
    <span
      className={clsx(
        "inline-flex items-center gap-1 rounded-full border px-2 py-0.5 font-mono text-[11px] font-medium",
        styles[variant],
      )}
    >
      {children}
    </span>
  );
}

export function Btn({
  children,
  onClick,
  disabled,
  variant = "primary",
}: {
  children: ReactNode;
  onClick?: () => void;
  disabled?: boolean;
  variant?: "primary" | "ghost" | "danger";
}) {
  const styles = {
    primary:
      "bg-accent-grad text-[#04122b] shadow-[0_4px_14px_-4px_rgba(91,157,255,0.55)] hover:brightness-110",
    ghost:
      "border border-line bg-panel2/60 text-text hover:border-line-strong hover:bg-panel2",
    danger:
      "bg-bad text-[#1a0202] shadow-[0_4px_14px_-4px_rgba(248,81,73,0.5)] hover:brightness-110",
  };
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      className={clsx(
        "rounded-lg px-3.5 py-2 text-[13px] font-semibold transition-all duration-150 active:scale-[0.98] disabled:cursor-not-allowed disabled:opacity-50 disabled:active:scale-100",
        styles[variant],
      )}
    >
      {children}
    </button>
  );
}

export function Label({ children }: { children: ReactNode }) {
  return (
    <label className="mb-1 mt-3 block text-[11px] font-medium uppercase tracking-wide text-muted">
      {children}
    </label>
  );
}

export function Input({
  value,
  onChange,
  placeholder,
}: {
  value: string;
  onChange: (v: string) => void;
  placeholder?: string;
}) {
  return (
    <input
      type="text"
      value={value}
      onChange={(e) => onChange(e.target.value)}
      placeholder={placeholder}
      className="w-full rounded-lg border border-line bg-panel2 px-3 py-2 text-[13px] text-text placeholder:text-muted/60 transition-colors hover:border-line-strong focus:border-accent focus:outline-none"
    />
  );
}

export function Textarea({
  value,
  onChange,
  placeholder,
  rows = 3,
  onKeyDown,
}: {
  value: string;
  onChange: (v: string) => void;
  placeholder?: string;
  rows?: number;
  onKeyDown?: (e: KeyboardEvent<HTMLTextAreaElement>) => void;
}) {
  return (
    <textarea
      value={value}
      onChange={(e) => onChange(e.target.value)}
      placeholder={placeholder}
      rows={rows}
      onKeyDown={onKeyDown}
      className="w-full resize-none rounded-lg border border-line bg-panel2 px-3 py-2 text-[13px] text-text placeholder:text-muted/60 transition-colors hover:border-line-strong focus:border-accent focus:outline-none"
    />
  );
}

export function Select({
  value,
  onChange,
  options,
}: {
  value: string;
  onChange: (v: string) => void;
  options: { value: string; label: string }[];
}) {
  return (
    <select
      value={value}
      onChange={(e) => onChange(e.target.value)}
      className="w-full cursor-pointer rounded-lg border border-line bg-panel2 px-3 py-2 text-[13px] text-text transition-colors hover:border-line-strong focus:border-accent focus:outline-none"
    >
      {options.map((o) => (
        <option key={o.value} value={o.value}>
          {o.label}
        </option>
      ))}
    </select>
  );
}

export function Status({ children }: { children: ReactNode }) {
  return <div className="mt-2 min-h-4 text-[12px] text-muted">{children}</div>;
}

export function Pre({ children }: { children: string }) {
  return (
    <pre className="mt-2 overflow-auto whitespace-pre-wrap rounded-lg border border-line bg-panel2/70 p-3 font-mono text-[12px] leading-relaxed text-text/90">
      {children}
    </pre>
  );
}
