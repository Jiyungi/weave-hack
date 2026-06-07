import clsx from "clsx";
import { ReactNode } from "react";

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
        "rounded-[10px] border border-line bg-panel p-4",
        className,
      )}
    >
      <h2 className="mb-3 text-[13px] font-medium uppercase tracking-wider text-muted">
        {title}
        {badge && <span className="ml-2 inline-block">{badge}</span>}
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
    good: "border-[#1c3d24] bg-[#0e1f13] text-good",
    bad: "border-[#3d1c1c] bg-[#1f0e0e] text-bad",
    warn: "border-line bg-panel2 text-warn",
    muted: "border-line bg-panel2 text-muted",
  };
  return (
    <span
      className={clsx(
        "inline-block rounded-full border px-2 py-0.5 font-mono text-[11px]",
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
    primary: "bg-accent text-[#04122b]",
    ghost: "border border-line bg-transparent text-text",
    danger: "bg-bad text-[#1a0202]",
  };
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      className={clsx(
        "rounded-[7px] px-3 py-2 text-[13px] font-semibold disabled:cursor-not-allowed disabled:opacity-50",
        styles[variant],
      )}
    >
      {children}
    </button>
  );
}

export function Label({ children }: { children: ReactNode }) {
  return (
    <label className="mb-1 mt-2 block text-[12px] text-muted">{children}</label>
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
      className="w-full rounded-[7px] border border-line bg-panel2 px-2.5 py-2 text-[13px] text-text"
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
      className="w-full rounded-[7px] border border-line bg-panel2 px-2.5 py-2 text-[13px] text-text"
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
  return (
    <div className="mt-2 min-h-4 text-[12px] text-muted">{children}</div>
  );
}

export function Pre({ children }: { children: string }) {
  return (
    <pre className="mt-2 overflow-auto whitespace-pre-wrap rounded-[7px] border border-line bg-panel2 p-2.5 font-mono text-[12px]">
      {children}
    </pre>
  );
}
