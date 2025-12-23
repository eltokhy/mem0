/** @type {import('next').NextConfig} */
const nextConfig = {
  // NEW: app is served from /dashboard
  basePath: '/dashboard',
  assetPrefix: '/dashboard',

  eslint: {
    ignoreDuringBuilds: true,
  },
  typescript: {
    ignoreBuildErrors: true,
  },
  images: {
    unoptimized: true,
  },
};

export default nextConfig;
