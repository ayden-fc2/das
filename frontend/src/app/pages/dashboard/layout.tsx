// app/dashboard/layout.tsx
'use client';
import * as React from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import {
    ChevronLeft,
    Menu,
    Home,
    People,
    AddCircle, AlignHorizontalRight, CloudQueue, CrisisAlert, DoneAll, Logout,
} from '@mui/icons-material';
import {Box, Typography} from "@mui/material";

const spaceNavData = [
    { icon: <CloudQueue />, label: 'Joined Spaces', path: '/spaces' }, // 增删改自己管理的space， 查自己管理的/自己加入的
    { icon: <AddCircle />, label: 'New Space', path: '/new-space' }, // 查所有的Space并申请加入，可以根据Org筛选
    { icon: <DoneAll />, label: 'Requests', path: '/join-request' }, //批准/拒绝加入请求，查看自己的所有请求
];

const projectNavData = [
    { icon: <AlignHorizontalRight />, label: 'Analysis', path: '/project-analysis' }, // 核心分析
    { icon: <CrisisAlert />, label: 'Faults Tracing', path: '/faults-tracing' }, // 故障记录
];

const personalNavData = [
    { icon: <People />, label: 'User & Groups', path: '/user-info' }, // 个人中心和组织中心
    { icon: <Logout />, label: 'Logout', path: '/logout' }, // 退出登录
];

export default function DashboardLayout({ children }: { children: React.ReactNode }) {
    const [open, setOpen] = React.useState(true);
    const pathname = usePathname();

    const toggleCollapse = () => setOpen(!open);

    return (
        <Box className="flex min-h-screen bg-gray-100">
            {/* 左侧导航栏 */}
            <aside
                className={`fixed h-screen transition-all duration-300 ${
                    open ? 'w-64' : 'w-16'
                } bg-[rgb(5,30,52)] shadow-lg`}
            >
                <nav className="flex h-full flex-col overflow-y-auto overflow-x-hidden hide-scrollbar">
                    {/* Logo 和折叠按钮区域 */}
                    <Box className="flex h-14 items-center justify-between border-b border-gray-700 px-4">
                        <Typography
                            className={`flex items-center transition-opacity ${
                                open ? 'w-full opacity-100' : 'w-0 opacity-0'
                            }`}
                        >
                            {open && <span className="text-xl font-semibold text-white whitespace-nowrap">DAS Dashboard</span>}
                        </Typography>

                        <button
                            onClick={toggleCollapse}
                            className="p-1.5 text-gray-400 hover:text-white rounded-full hover:bg-gray-700 transition-colors"
                        >
                            {open ? (
                                <ChevronLeft className="w-6 h-6" />
                            ) : (
                                <Menu className="w-6 h-6" />
                            )}
                        </button>
                    </Box>

                    {/* 主菜单区域 */}
                    <Box className="flex-1 overflow-y-auto py-2 hide-scrollbar">
                        {/* 仪表盘入口 */}
                        <Link href="/pages/dashboard">
                            <Box
                                className={`mx-2 mb-1 p-2 rounded-lg flex items-center ${pathname==='/pages/dashboard' ? 'text-[rgb(102,157,246)]' : 'text-white'} transition-colors`}
                            >
                                <Home className={`w-6 h-6 ${open ? 'mr-3' : 'mx-auto'}`} />
                                {open && (
                                    <span className="text-sm font-medium truncate my-2">Project Overview</span>
                                )}
                            </Box>
                        </Link>

                        {/* 管理空间 */}
                        <Box className={` ${open ? 'bg-white/5' : ''} pt-2 pb-4`}>
                            {open && (
                                <Box className="px-3 pt-2 pb-4">
                                    <h3 className="text-sm font-semibold text-gray-400 mb-1">Manage Spaces</h3>
                                    <p className="text-xs text-gray-500 leading-relaxed">
                                        Create or join a workspace to manage DCS diagrams for specific projects
                                    </p>
                                </Box>
                            )}

                            {spaceNavData.map((item) => (
                                <Link
                                    key={item.path}
                                    href={`/pages/dashboard${item.path}`}
                                    className="block"
                                >
                                    <Box
                                        className={`mx-2 mb-1 p-2 rounded-lg flex items-center ${
                                            pathname === `/pages/dashboard${item.path}`
                                                ? 'bg-blue-100/10 text-blue-400'
                                                : 'text-gray-300 hover:bg-white/10 hover:text-white'
                                        } transition-colors`}
                                    >
                                        {React.cloneElement(item.icon, {
                                            className: `w-6 h-6 ${open ? 'mr-3' : 'mx-auto'}`
                                        })}
                                        {open && (
                                            <span className="text-sm font-medium truncate">{item.label}</span>
                                        )}
                                    </Box>
                                </Link>
                            ))}
                        </Box>

                        {/* 项目分析 */}
                        <Box className={` ${open ? 'bg-white/5' : ''} pt-2 pb-4`}>
                            {open && (
                                <div className="px-3 pt-2 pb-4">
                                    <h3 className="text-sm font-semibold text-gray-400 mb-1">Project Analysis</h3>
                                    <p className="text-xs text-gray-500 leading-relaxed">
                                        Project analysis and fault tracing
                                    </p>
                                </div>
                            )}

                            {projectNavData.map((item) => (
                                <Link
                                    key={item.path}
                                    href={`/pages/dashboard${item.path}`}
                                    className="block"
                                >
                                    <Box
                                        className={`mx-2 mb-1 p-2 rounded-lg flex items-center ${
                                            pathname === `/pages/dashboard${item.path}`
                                                ? 'bg-blue-100/10 text-blue-400'
                                                : 'text-gray-300 hover:bg-white/10 hover:text-white'
                                        } transition-colors`}
                                    >
                                        {React.cloneElement(item.icon, {
                                            className: `w-6 h-6 ${open ? 'mr-3' : 'mx-auto'}`
                                        })}
                                        {open && (
                                            <span className="text-sm font-medium truncate">{item.label}</span>
                                        )}
                                    </Box>
                                </Link>
                            ))}
                        </Box>

                        {/* 个人中心 */}
                        <Box className={` ${open ? 'bg-white/5' : ''} pt-2 pb-4`}>
                            {open && (
                                <div className="px-3 pt-2 pb-4">
                                    <h3 className="text-sm font-semibold text-gray-400 mb-1">Personal Center</h3>
                                    <p className="text-xs text-gray-500 leading-relaxed">
                                        Personal profile, organizations and managing login status

                                    </p>
                                </div>
                            )}

                            {personalNavData.map((item) => (
                                <Link
                                    key={item.path}
                                    href={`/pages/dashboard${item.path}`}
                                    className="block"
                                >
                                    <div
                                        className={`mx-2 mb-1 p-2 rounded-lg flex items-center ${
                                            pathname === `/pages/dashboard${item.path}`
                                                ? 'bg-blue-100/10 text-blue-400'
                                                : 'text-gray-300 hover:bg-white/10 hover:text-white'
                                        } transition-colors`}
                                    >
                                        {React.cloneElement(item.icon, {
                                            className: `w-6 h-6 ${open ? 'mr-3' : 'mx-auto'}`
                                        })}
                                        {open && (
                                            <span className="text-sm font-medium truncate">{item.label}</span>
                                        )}
                                    </div>
                                </Link>
                            ))}
                        </Box>
                    </Box>
                </nav>
            </aside>

            {/* 内容区域 */}
            <main
                className={`flex-1 flex h-[100vh] bg-gray-100 relative transition-margin duration-300 ${
                    open ? 'ml-64' : 'ml-16'
                }`}
            >
                <Box className="m-4 bg-transparent flex-1 text-gray-800">
                    {children}
                </Box>
            </main>
        </Box>
    );
}