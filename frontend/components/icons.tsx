import type { ReactNode, SVGProps } from "react";

type IconProps = SVGProps<SVGSVGElement> & { size?: number };

function Icon({ size = 16, children, ...props }: IconProps & { children: ReactNode }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={2}
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      focusable="false"
      {...props}
    >
      {children}
    </svg>
  );
}

export const PlayIcon = (p: IconProps) => (
  <Icon {...p}><path d="M7 4.5v15l12.5-7.5z" fill="currentColor" stroke="none" /></Icon>
);
export const PauseIcon = (p: IconProps) => (
  <Icon {...p}><rect x="6" y="4.5" width="4" height="15" rx="1" fill="currentColor" stroke="none" /><rect x="14" y="4.5" width="4" height="15" rx="1" fill="currentColor" stroke="none" /></Icon>
);
export const PrevIcon = (p: IconProps) => (
  <Icon {...p}><path d="M18 6 10 12l8 6z" fill="currentColor" stroke="none" /><path d="M6 6v12" /></Icon>
);
export const NextIcon = (p: IconProps) => (
  <Icon {...p}><path d="m6 6 8 6-8 6z" fill="currentColor" stroke="none" /><path d="M18 6v12" /></Icon>
);
export const ChevronLeft = (p: IconProps) => <Icon {...p}><path d="m15 18-6-6 6-6" /></Icon>;
export const ChevronRight = (p: IconProps) => <Icon {...p}><path d="m9 18 6-6-6-6" /></Icon>;
export const ArrowRight = (p: IconProps) => <Icon {...p}><path d="M5 12h14M13 6l6 6-6 6" /></Icon>;
export const ArrowUpRight = (p: IconProps) => <Icon {...p}><path d="M7 17 17 7M7 7h10v10" /></Icon>;
export const UploadIcon = (p: IconProps) => (
  <Icon {...p}><path d="M12 15V4M7 9l5-5 5 5" /><path d="M4 15v3a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-3" /></Icon>
);
export const TrashIcon = (p: IconProps) => (
  <Icon {...p}><path d="M4 7h16M10 11v6M14 11v6M6 7l1 12a2 2 0 0 0 2 2h6a2 2 0 0 0 2-2l1-12M9 7V4h6v3" /></Icon>
);
export const AlertIcon = (p: IconProps) => (
  <Icon {...p}><path d="M12 3 2 20h20z" /><path d="M12 10v4M12 17h.01" /></Icon>
);
export const InfoIcon = (p: IconProps) => (
  <Icon {...p}><circle cx="12" cy="12" r="9" /><path d="M12 11v5M12 8h.01" /></Icon>
);
export const OfflineIcon = (p: IconProps) => (
  <Icon {...p}><path d="M2 2l20 20M8.5 16.5a5 5 0 0 1 7 0M5 13a10 10 0 0 1 5.17-2.8M19 13a10 10 0 0 0-2.3-1.6M2 8.8a15 15 0 0 1 4.2-2.6M22 8.8A15 15 0 0 0 10.7 5M12 20h.01" /></Icon>
);
export const ZoomIn = (p: IconProps) => <Icon {...p}><circle cx="11" cy="11" r="7" /><path d="m20 20-4-4M11 8v6M8 11h6" /></Icon>;
export const ZoomOut = (p: IconProps) => <Icon {...p}><circle cx="11" cy="11" r="7" /><path d="m20 20-4-4M8 11h6" /></Icon>;
export const MicIcon = (p: IconProps) => (
  <Icon {...p}><rect x="9" y="3" width="6" height="11" rx="3" /><path d="M5 11a7 7 0 0 0 14 0M12 18v3" /></Icon>
);
export const StopIcon = (p: IconProps) => (
  <Icon {...p}><rect x="6" y="6" width="12" height="12" rx="2" fill="currentColor" stroke="none" /></Icon>
);
export const CheckIcon = (p: IconProps) => <Icon {...p}><path d="m5 12 5 5L20 7" /></Icon>;
export const XIcon = (p: IconProps) => <Icon {...p}><path d="M6 6l12 12M18 6 6 18" /></Icon>;
export const LockIcon = (p: IconProps) => (
  <Icon {...p}><rect x="5" y="11" width="14" height="10" rx="2" /><path d="M8 11V7a4 4 0 0 1 8 0v4" /></Icon>
);
export const RefreshIcon = (p: IconProps) => (
  <Icon {...p}><path d="M20 11a8 8 0 1 0-2.3 5.7M20 4v7h-7" /></Icon>
);
export const WaveIcon = (p: IconProps) => (
  <Icon {...p}><path d="M3 12h2M7 8v8M11 4v16M15 7v10M19 10v4M21 12h0" /></Icon>
);
export const ChartIcon = (p: IconProps) => (
  <Icon {...p}><path d="M4 20V4M4 20h16M8 16l4-5 3 3 5-7" /></Icon>
);
export const SearchIcon = (p: IconProps) => <Icon {...p}><circle cx="11" cy="11" r="7" /><path d="m20 20-4-4" /></Icon>;
export const VolumeIcon = (p: IconProps) => (
  <Icon {...p}><path d="M4 9v6h4l5 4V5L8 9zM16 9a4 4 0 0 1 0 6M18.5 6.5a8 8 0 0 1 0 11" /></Icon>
);
export const DatabaseIcon = (p: IconProps) => (
  <Icon {...p}><ellipse cx="12" cy="5" rx="8" ry="3" /><path d="M4 5v14c0 1.7 3.6 3 8 3s8-1.3 8-3V5M4 12c0 1.7 3.6 3 8 3s8-1.3 8-3" /></Icon>
);

export function BrandMark({ size = 26 }: { size?: number } = {}) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" aria-hidden="true" style={{ flexShrink: 0 }}>
      <rect x="2" y="10" width="2.4" height="4" rx="1.2" fill="#3987e5" />
      <rect x="6.2" y="6" width="2.4" height="12" rx="1.2" fill="#199e70" />
      <rect x="10.4" y="2.5" width="2.4" height="19" rx="1.2" fill="#d55181" />
      <rect x="14.6" y="7" width="2.4" height="10" rx="1.2" fill="#c98500" />
      <rect x="18.8" y="9.5" width="2.4" height="5" rx="1.2" fill="#d95926" />
    </svg>
  );
}
