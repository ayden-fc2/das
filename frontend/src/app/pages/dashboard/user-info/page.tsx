"use client"

import React, {useEffect} from "react";
import {
    Avatar,
    Box,
    Button,
    Dialog, DialogActions,
    DialogContent,
    DialogContentText,
    DialogTitle,
    Divider, Pagination, Paper, Table, TableBody, TableCell, TableContainer, TableHead, TableRow, TextField,
    Typography
} from "@mui/material";
import {useAuth} from "@/app/context/AuthContext";
import CalendarHeatmap from 'react-calendar-heatmap';
import 'react-calendar-heatmap/dist/styles.css';
import {err, success} from "@/app/utils/alerter";
import {
    createOrg,
    deleteOrg,
    getMyOrgNum,
    getMyOrgs,
    joinOrg,
    quitOrgByUsr,
    updateOrgCode,
    updateOrgInfo
} from "@/app/api/org";
import {Stack} from "@mui/system";
import {convertToChinaTime} from "@/app/utils/common";

const UserInfo: React.FC = () => {
    const authContext = useAuth()
    // 计算 startDate（当前日期的 7 个月前的 1 号）
    const startDate = new Date();
    startDate.setMonth(startDate.getMonth() - 7);
    startDate.setDate(1);

    // 计算 endDate（下个月的 1 号）
    const endDate = new Date();
    endDate.setMonth(endDate.getMonth() + 1);
    endDate.setDate(1);

    /**
     * 弹窗新增组织
     */
    const [open, setOpen] = React.useState(false);
    const [newOrgName, setNewOrgName] = React.useState("");
    const [newOrgDescription, setNewOrgDescription] = React.useState("");
    const handleClickOpen = () => {
        setOpen(true);
    };

    const handleClose = () => {
        setOpen(false);
    };

    const handleCreate = () => {
        if (newOrgName.length < 0 || newOrgName.length > 20) {
            err('Org name must be between 1 and 20 characters');
            return
        }
        createOrg(newOrgName, newOrgDescription).then(res=> {
            if (res.success) {
                success('Successfully created');
                handleClose();
                setNewOrgName("");
                setNewOrgDescription("");
                refreshMyOrgs()
                return
            }
            err(res.message);
        }).catch(e=> {
            console.error(e);
            err(e)
        })
    }

    /**
     * 弹窗加入组织
     */
    const [openJoin, setOpenJoin] = React.useState(false);
    const [orgCode, setOrgCode] = React.useState("");
    const handleJoinOpen = () => {
        setOpenJoin(true);
    }
    const handleJoinClose =() => {
        setOpenJoin(false);
    }
    const handleJoinOrg = () => {
        if (orgCode.length < 0) {
            err('Wrong org code')
            return
        }
        joinOrg(orgCode).then(res=> {
            if (res.success) {
                success('Successfully joined');
                handleJoinClose()
                setOrgCode("");
                refreshMyOrgs()
                return
            }
            err(res.message);
        }).catch(e=> {
            console.error(e);
            err(e)
        })
    }


    /**
     * 查询用户相关组织
     */
    const [page, setPage] = React.useState(1);
    const [size, setSize] = React.useState(10);
    const [total, setTotal] = React.useState(0);
    const [currentTableRows, setCurrentTableRows] = React.useState([]);
    const refreshMyOrgs = () => {
        getMyOrgNum().then(res=> {
            if (res.success) {
                setTotal(res.data);
                return
            }
            err(res.message);
        }).catch(e=> {
            console.error(e);
            err(e)
        })
        getMyOrgs(page, size).then(res=> {
            if (res.success) {
                setCurrentTableRows(res.data);
                return
            }
            err(res.message);
        }).catch(e => {
            console.error(e);
            err(e)
        })
    }
    const handleRefresh = (e:any, newPage: number) => {
        setPage(newPage);
        refreshMyOrgs()
    }

    const quitOrg = (orgId: number) => {
        const confirmed = window.confirm("Are you sure you want to quit this organization?");
        if (confirmed) {
            console.log("Quit org:", orgId);
            quitOrgByUsr(orgId).then(res => {
                if (res.success) {
                    refreshMyOrgs()
                    return
                }
                err(res.message);
            }).catch(e=> {
                console.error(e);
            })
        }
    }

    /**
     * 管理组织
     */
    const [showManage, setShowManage] = React.useState(false);
    const [currentOrgId, setCurrentOrgId] = React.useState(0);
    const [currentOrgName, setCurrentOrgName] = React.useState("");
    const [currentOrgDesc, setCurrentOrgDesc] = React.useState("");
    const [currentOrgCode, setCurrentOrgCode] = React.useState("");
    const handleOpenManage = () => {
        setShowManage(true);
    }
    const handleCloseManage = () => {
        setShowManage(false);
        setCurrentOrgId(0);
        setCurrentOrgName("");
        setCurrentOrgDesc("");
        setCurrentOrgCode("");
        refreshMyOrgs()
    }
    const manageOrg = (orgId: number, orgName: string, orgDesc: string, orgCode: string) => {
        handleOpenManage()
        setCurrentOrgId(orgId)
        setCurrentOrgName(orgName)
        setCurrentOrgDesc(orgDesc)
        setCurrentOrgCode(orgCode)
    }
    const updateCurrentOrg = () => {
        if (currentOrgName.length < 0 || currentOrgName.length > 20) {
            err('Org name must be between 1 and 20 characters');
            return
        }
        updateOrgInfo(currentOrgId, currentOrgName, currentOrgDesc).then(res => {
            if (res.success) {
                success('Successfully updated');
                return
            }
            err(res.message);
        }).catch(e => {
            console.error(e);
        })
    }
    const handleUpdateOrgCode = () => {
        updateOrgCode(currentOrgId).then(res => {
            if (res.success) {
                success('Successfully updated');
                setCurrentOrgCode(res.data);
                return
            }
        }).catch(e => {
            console.error(e);
        })
    }

    const handleDeleteOrg = () => {
        deleteOrg(currentOrgId).then(res=> {
            if (res.success) {
                success('Successfully deleted');
                handleCloseManage();
                return
            }
            err(res.message);
        }).catch(e=> {
            console.error(e);
        })
    }


    /**
     * 全局
     */
    const handleCopy = async (content: string) => {
        try {
            await navigator.clipboard.writeText(content);
            success('Successfully copied');
        } catch (err) {
            console.error('复制失败:', err);
        }
    }
    const roleMap: Record<string, { label: string; color: string }> = {
        '1': { label: 'Operator', color: '#1890ff' },       // 蓝色
        '2': { label: 'Engineer', color: '#52c41a' },       // 绿色
        '3': { label: 'Admin', color: '#faad14' },          // 黄色
        '4': { label: 'Super Admin', color: '#f5222d' },    // 红色
    };
    const getRoleTags = (ids: string) => {
        const idTable = ids.split(',');
        return idTable.map(id => {
            const role = roleMap[id];
            return role ? (
                <span key={id} style={{
                    backgroundColor: role.color,
                    color: '#fff',
                    padding: '2px 8px',
                    borderRadius: '12px',
                    fontSize: '12px',
                    marginRight: '6px',
                    display: 'inline-block',
                }}>
        {role.label}
      </span>
            ) : null;
        });
    }
    const checkManager = (ids: string) => {
        const idTable = ids.split(',');
        return idTable.some(id => id === '3')
    }
    useEffect(()=> {
        refreshMyOrgs()
    }, [])

    return (
        <Box className={`flex flex-col h-full w-full p-4`}>
            {/*  用户信息（昵称更改），信息图表统计  */}
            <Box className={`w-full h-36 p-4 bg-white shadow rounded-lg flex`}>
                <Box className={`h-full w-96 relative flex items-center justify-between`}>
                    {/*  头像  */}
                    <Avatar className={`!h-20 !w-20`} alt="Remy Sharp" src={authContext.basicUserInfo?.avatar || ''} />
                    {/*  昵称和邮箱  */}
                    <Box className={`h-full flex-1 ml-4 flex-col flex justify-center relative`}>
                        <Typography variant="h6" className={`font-bold max-w-60 truncate whitespace-nowrap overflow-hidden`}>{authContext.basicUserInfo?.nickname || 'Default Name'}</Typography>
                        <Typography variant="body2" className={`text-gray-600 truncate max-w-60 whitespace-nowrap overflow-hidden`}>{authContext.basicUserInfo?.email || ''}</Typography>
                    </Box>
                </Box>
                <Divider />
                {/*  github热力图  */}
                <Box className={`flex-1 ml-4`}>
                    <Box className={`flex items-center justify-end`}>
                        <CalendarHeatmap
                            startDate={startDate}
                            endDate={endDate}
                            values={[
                                { date: '2016-01-01', count: 12 },
                                { date: '2016-01-22', count: 122 },
                                { date: '2016-01-30', count: 38 },
                                // TODO 获取打点统计
                            ]}
                        />
                    </Box>
                </Box>
            </Box>
            {/*  我的组织以及角色，允许新建组织、管理自己的组织和角色  */}
            <Box className={`w-full flex-1 mt-4 px-4 py-2 bg-white shadow rounded-lg flex flex-col`}>
                <Box className={`w-full h-12 flex items-center justify-between mt-2`}>
                    <Typography variant={`h6`}>My Organizations</Typography>
                    <Box>
                        <Button onClick={handleClickOpen} className={`!mx-4`} variant={`contained`} color="primary">Create Organization</Button>
                        <Button onClick={handleJoinOpen} variant={`contained`} color="success">Join Organization</Button>
                    </Box>
                </Box>
                <Stack className={`my-4 flex flex-col items-center h-full`} spacing={2}>
                    <TableContainer className={`w-full flex-1`}>
                        <Table>
                            <TableHead>
                                <TableRow>
                                    <TableCell align={`center`}>Name</TableCell>
                                    <TableCell align={`center`}>Description</TableCell>
                                    <TableCell align={`center`}>Owner</TableCell>
                                    <TableCell align={`center`}>Created Time</TableCell>
                                    <TableCell align={`center`}>Code</TableCell>
                                    <TableCell align={`center`}>Roles</TableCell>
                                    <TableCell align={`center`}>Operation</TableCell>
                                </TableRow>
                            </TableHead>
                            <TableBody>
                                {
                                    currentTableRows.map((row: any, index) => (
                                        <TableRow key={index}>
                                            <TableCell align={`center`}>{row.org_name}</TableCell>
                                            <TableCell align={`center`}>{row.org_desc}</TableCell>
                                            <TableCell align={`center`}>
                                                <span className={`ml-2 text-blue-600 cursor-pointer`} onClick={() => {handleCopy(row.phoneNum)}}>{row.nickName}</span>
                                            </TableCell>
                                            <TableCell align={`center`}>{convertToChinaTime(row.created_time)}</TableCell>
                                            <TableCell align={`center`}>
                                                <span className={`ml-2 text-blue-600 cursor-pointer`} onClick={() => {handleCopy(row.org_code)}}>copy</span>
                                            </TableCell>
                                            <TableCell align={`center`}>
                                                {getRoleTags(row.authorityIds)}
                                            </TableCell>
                                            <TableCell align={`center`}>
                                                {
                                                    checkManager(row.authorityIds) ?
                                                        <Box className={`flex flex-col`}>
                                                            <Button onClick={()=>{manageOrg(row.org_id, row.org_name, row.org_desc, row.org_code)}}>Manage Org</Button>
                                                            <Button>Manage Members</Button>
                                                        </Box> :
                                                        <Button onClick={() => {quitOrg(row.org_id)}}>Quit</Button>}
                                            </TableCell>
                                        </TableRow>
                                    ))
                                }
                            </TableBody>
                        </Table>
                    </TableContainer>
                    <Pagination className={`flex items-center justify-center`} count={Math.ceil(total/size)} page={page} onChange={handleRefresh} />
                </Stack>
                {/*  筛选表格组件，组织名称、管理员、创建日期、用户角色，可选操作是管理组织弹窗  */}
                {/*  按钮弹窗新建组织（复用） */}
                <Dialog
                    open={open}
                    onClose={handleClose}
                    aria-labelledby="alert-dialog-title"
                    aria-describedby="alert-dialog-description"
                >
                    <DialogTitle id="alert-dialog-title">
                        {"Create a New Organization"}
                    </DialogTitle>
                    <DialogContent className={`flex flex-col`}>
                        <TextField onChange={(e) => setNewOrgName(e.target.value)} value={newOrgName} className={`!my-2 w-96`} id="outlined-basic" label="Org Name" size={`small`} variant="outlined" />
                        <TextField onChange={(e) => setNewOrgDescription(e.target.value)} value={newOrgDescription} className={`!my-2 w-96`} id="outlined-basic" label="Org Description" size={`small`} variant="outlined" />
                    </DialogContent>
                    <DialogActions>
                        <Button color={`error`} onClick={handleClose} autoFocus>
                            Cancel
                        </Button>
                        <Button onClick={handleCreate}>Create</Button>
                    </DialogActions>
                </Dialog>
                {/*  按钮加入组织  */}
                <Dialog
                    open={openJoin}
                    onClose={handleJoinClose}
                    aria-labelledby="alert-dialog-title"
                    aria-describedby="alert-dialog-description"
                >
                    <DialogTitle id="alert-dialog-title">
                        {"Join a Existing Organization"}
                    </DialogTitle>
                    <DialogContent className={`flex flex-col`}>
                        <TextField onChange={(e) => setOrgCode(e.target.value)} value={orgCode} className={`!my-2 w-96`} id="outlined-basic" label="Org Code" size={`small`} variant="outlined" />
                    </DialogContent>
                    <DialogActions>
                        <Button color={`error`} onClick={handleJoinClose} autoFocus>
                            Cancel
                        </Button>
                        <Button onClick={handleJoinOrg}>Join</Button>
                    </DialogActions>
                </Dialog>
                {/*  管理组织  */}
                <Dialog
                    open={showManage}
                    onClose={handleCloseManage}
                    aria-labelledby="alert-dialog-title"
                    aria-describedby="alert-dialog-description">
                    <DialogTitle id="alert-dialog-title">
                        {`Manage Org Basic Information`}
                    </DialogTitle>
                    <DialogContent className={`flex flex-col items-center`}>
                        {/*  更新组织信息， 删除群组， 查询所有用户信息， 更改用户角色， 更改组织代码  */}
                        <Box className={`flex items-center`}>
                            <Box className={`flex flex-col w-96 h-28 justify-around`}>
                                <TextField onChange={(e) => setCurrentOrgName(e.target.value)} value={currentOrgName} id="outlined-basic" label="Org Name" size={`small`} variant="outlined" />
                                <TextField onChange={(e) => setCurrentOrgDesc(e.target.value)} value={currentOrgDesc} id="outlined-basic" label="Org Description" size={`small`} variant="outlined" />
                            </Box>
                            <Button onClick={updateCurrentOrg} className={`!ml-4`}>Update</Button>
                        </Box>
                        <Box className={`flex items-center`}>
                            <Typography variant={`body1`} className={`w-96`}><strong>Code: </strong>{currentOrgCode}</Typography>
                            <Button onClick={handleUpdateOrgCode} className={`!ml-4`}>Update</Button>
                        </Box>
                        <Button onClick={handleDeleteOrg} className={`w-full !mt-2`} color={`error`} variant={`outlined`}>Delete</Button>
                    </DialogContent>
                </Dialog>
            </Box>
        </Box>
    );
};

export default UserInfo;
