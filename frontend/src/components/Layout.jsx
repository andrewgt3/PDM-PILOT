import React from 'react';
import { LayoutDashboard, Server, Settings, Activity, Cpu } from 'lucide-react';
import { Link, useLocation } from 'react-router-dom';

const SidebarItem = ({ icon: Icon, label, path, active }) => (
  <Link
    to={path}
    className={`flex items-center gap-3 px-3 py-2 rounded-md transition-colors text-sm font-medium ${active
      ? 'bg-slate-800 text-white border border-slate-700'
      : 'text-slate-400 hover:bg-slate-800 hover:text-slate-200'
      }`}
    style={{ textDecoration: 'none' }}
  >
    <Icon size={18} />
    <span>{label}</span>
  </Link>
);

const Layout = ({ children }) => {
  const location = useLocation();

  return (
    <div className="flex bg-[var(--bg-app)] min-h-screen">
      {/* Sidebar */}
      <aside
        className="w-64 border-r border-[var(--border-subtle)] bg-[var(--bg-app)] flex flex-col"
        style={{ position: 'sticky', top: 0, height: '100vh' }}
      >
        <div className="h-16 flex items-center px-6 border-b border-[var(--border-subtle)]">
          <div className="flex items-center gap-2 font-bold text-white tracking-wider">
            <Cpu size={20} className="text-[var(--accent-primary)]" />
            <span>PREDICT<span className="text-[var(--text-secondary)]">LAB</span></span>
          </div>
        </div>

        <nav className="flex-1 p-4 space-y-1">
          <div className="text-xs font-bold text-slate-500 uppercase tracking-wider mb-4 px-2">Platform</div>
          <SidebarItem
            icon={LayoutDashboard}
            label="Overview"
            path="/"
            active={location.pathname === '/' || location.pathname === '/plant'}
          />
          <SidebarItem
            icon={Server}
            label="Asset Monitor"
            path="/assets"
            active={location.pathname.startsWith('/assets') || location.pathname === '/cell'}
          />
          <SidebarItem
            icon={Activity}
            label="Model Performance"
            path="/audit"
            active={location.pathname === '/audit' || location.pathname === '/models'}
          />
        </nav>

        <div className="p-4 border-t border-[var(--border-subtle)]">
          <div className="flex items-center gap-3 px-2">
            <div className="w-8 h-8 rounded-full bg-slate-700 flex items-center justify-center text-xs font-bold text-slate-300">
              AG
            </div>
            <div className="flex flex-col">
              <span className="text-xs font-bold text-white">Guest User</span>
              <span className="text-[10px] text-slate-500">Reliability Engineer</span>
            </div>
          </div>
        </div>
      </aside>

      {/* Main Content */}
      <main className="flex-1 flex flex-col min-w-0">
        <header className="h-16 border-b border-[var(--border-subtle)] flex items-center justify-between px-8 bg-[var(--bg-app)]/80 backdrop-blur">
          <div className="text-sm breadcrumbs text-slate-400">
            <span className="opacity-50">Platform</span> / <span className="text-white font-medium">
              {location.pathname === '/' ? 'Overview' :
                location.pathname.startsWith('/assets') ? 'Asset Monitor' : 'Model Intel'}
            </span>
          </div>
          <div className="flex items-center gap-4">
            <span className="flex items-center gap-2 text-xs font-mono text-slate-500 bg-slate-900 border border-slate-800 px-2 py-1 rounded">
              <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></span>
              SYSTEM ONLINE
            </span>
          </div>
        </header>

        <div className="flex-1 p-8 overflow-auto">
          {children}
        </div>
      </main>
    </div>
  );
};

export default Layout;
