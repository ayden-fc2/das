package com.auth.oauth2.service.impl;

import com.alibaba.fastjson.JSON;
import com.alibaba.fastjson.JSONObject;
import com.auth.oauth2.entity.TmpInfo;
import com.auth.oauth2.service.*;
import com.example.common.dto.ResponseBean;
import com.auth.oauth2.entity.AccountSt;
import com.auth.oauth2.entity.RelationshipSt;
import com.example.common.enums.UserType;
import com.example.common.exception.MyException;
import com.auth.oauth2.mapper.UserMapper;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import redis.clients.jedis.Jedis;


import javax.annotation.Resource;
import java.util.*;

@Service
public class SignServiceImpl implements SignService {

    @Autowired
    FeignService fs;

    @Autowired
    UserMapper um;

    @Autowired
    private TokenFeignServiceDocker dockerService;

    @Autowired
    private TokenFeignServiceLocal localService;

    public String getToken(Map<String, String> request) {
        try {
            // 尝试使用 Docker 内部地址调用
            return dockerService.getToken(request);
        } catch (Exception e) {
            System.out.println("连接 Docker auth-service 失败，尝试 localhost...");
            // 调用失败后，使用本地地址调用
            return localService.getToken(request);
        }
    }

    //api 登录
    @Override
    public ResponseBean signInCheck(String tryPhoneNum, String tryPassword) {
        try {
            List<AccountSt> accountList = um.selectAccounts(tryPhoneNum);
            if (accountList.size()<1){
                return ResponseBean.fail("登陆失败，该账号还未注册");
            }else{
                if (accountList.get(0).getPasswordDetail().equals(tryPassword)){
                    //请求某用户的OAuth2令牌
                    String JWTToken = getJWTByUser(accountList.get(0));
                    return ResponseBean.success(JWTToken);
                }else {
                    return ResponseBean.fail("密码错误");
                }
            }
        }catch (Exception e){
            throw new MyException("登陆失败，错误："+e);
        }
    }

    //向OAuth2获取JWT令牌
    private String getJWTByUser(AccountSt accountSt){
        //构建请求
        Map<String, String> querys = new HashMap<String, String>();
        querys.put("grant_type", "client_credentials");
        querys.put("client_id", "zuul_server");
        querys.put("client_secret", "secret");
        querys.put("scope", "write read");
        //获取权限列表
        long userId = accountSt.getAccountId();
        List<RelationshipSt> relationshipStList = um.selectRelations(userId);

//        String isSuperManager = "0";
//        String isManager = "0";
//        String isController = "0";
//        String isObserver = "0";
//        for (RelationshipSt rs :
//                relationshipStList) {
//            switch ((int) rs.getAuthorityId()){
//                case 4:
//                    isSuperManager = "1";
//                    break;
//                case 3:
//                    isManager = "1";
//                    break;
//                case 2:
//                    isController = "1";
//                    break;
//                case 1:
//                    isObserver = "1";
//                    break;
//            }
//        }
//        //
//        querys.put("isSuperManager", isSuperManager);
//        querys.put("isManager", isManager);
//        querys.put("isController", isController);
//        querys.put("isObserver", isObserver);

        // 构建组织-权限映射
        // key：组织ID；value：角色名称（如 observer, controller, manager, superManager）
        Map<String, List<String>> orgRoleMap = new HashMap<>();

        for (RelationshipSt rs : relationshipStList) {
            String orgId = String.valueOf(rs.getOrgId());
            String role = "";
            switch ((int) rs.getAuthorityId()) {
                case 4:
                    role = UserType.SuperManager.getType();
                    break;
                case 3:
                    role = UserType.Manager.getType();
                    break;
                case 2:
                    role = UserType.Controller.getType();
                    break;
                case 1:
                    role = UserType.Observer.getType();
                    break;
                default:
                    role = null;
            }
            if (!orgRoleMap.containsKey(orgId)) {
                orgRoleMap.put(orgId, new ArrayList<String>());
                orgRoleMap.get(orgId).add(role);
            } else {
                orgRoleMap.get(orgId).add(role);
            }
        }

        // 将组织-权限映射转换为JSON字符串，并加入请求参数
        querys.put("orgRoles", JSONObject.toJSONString(orgRoleMap));

        querys.put("userId", String.valueOf(userId));
        String response = getToken(querys);
        JSONObject jsonObject = JSONObject.parseObject(response);
        return jsonObject.getString("access_token");
    }

    public static Jedis getJedisConnection() {
        String redisHost = "redis"; // Docker 内部默认使用服务名
        int redisPort = 6379;

        try {
            Jedis jedis = new Jedis(redisHost, redisPort);
            jedis.ping(); // 发送 Ping 确保 Redis 可用
            return jedis;
        } catch (Exception e) {
            System.out.println("连接 Docker Redis 失败，尝试 localhost...");
            return new Jedis("localhost", redisPort); // 本地环境使用 localhost
        }
    }
    //api 获取验证码
    @Override
    public ResponseBean getPhoneCode(String phoneNum, int mode) {
        //生成6位验证码
        String verificationCode = generateVerificationCode(6);
        // 创建一个 VerificationInfo 对象并设置账号、验证码和模式
        TmpInfo info = new TmpInfo();
        info.setAccountNumber(phoneNum);
        info.setVerificationCode(verificationCode);
        info.setMode(phoneNum+mode);
        // 将 VerificationInfo 对象转换为 JSON 字符串
        String jsonInfo = JSON.toJSONString(info);

        // 暂时不发送验证码 TODO
        // if (sendCode(phoneNum,verificationCode)){
        if (true){
            try {
                // 连接 Redis
                // Jedis jedis = new Jedis("localhost", 6379);
                Jedis jedis = getJedisConnection();
                // 将信息存储到 Redis 中，使用mode作为键
                jedis.setex(info.getMode(), 120, jsonInfo);
                System.out.println(jsonInfo);
                // 关闭 Redis 连接
                jedis.close();
                return ResponseBean.success();
            }catch (Exception e){
                throw new MyException("获取验证码时服务器调取redis失败，错误："+e);
            }
        }
        return ResponseBean.fail("发送验证码失败");
    }
    //生成随机验证码
    public static String generateVerificationCode(int length) {
        String numbers = "0123456789"; // 可供选择的字符集合

        Random random = new Random();
        StringBuilder code = new StringBuilder(length);

        for (int i = 0; i < length; i++) {
            int randomIndex = random.nextInt(numbers.length());
            code.append(numbers.charAt(randomIndex));
        }
        return code.toString();
    }
    //发送验证码
    private boolean sendCode(String phoneNum, String code){
        String appcode = " 3d65f8435f184453bc65d9269585e979";

        Map<String, String> querys = new HashMap<String, String>();
        querys.put("mobile", phoneNum);
        querys.put("param", "**code**:"+code+",**minute**:1");
        //smsSignId（短信前缀）和templateId（短信模板），可登录国阳云控制台自助申请。参考文档：http://help.guoyangyun.com/Problem/Qm.html
        querys.put("smsSignId", "2e65b1bb3d054466b82f0c9d125465e2");
        querys.put("templateId", "908e94ccf08b4476ba6c876d13f084ad");

        String authorization = "APPCODE " + appcode;

        try {
            String response = fs.sendCode(authorization,querys);
            //获取response的code
            JSONObject jsonObject = JSONObject.parseObject(response);
            String codeValue = jsonObject.getString("code");
            System.out.println("发送短信验证码返回："+jsonObject.toJSONString());
            if (codeValue.equals("0")){
                return true;
            }else {
                return false;
            }
        } catch (Exception e) {
            throw new MyException("发送验证码失败，错误"+e);
        }
    }
    //获取redis里的验证码
    public static String checkCode(String phoneNum, int mode){
        Jedis jedis = new Jedis("localhost", 6379);
        String code = "none"; // 默认返回值

        // 获取存储的 JSON 字符串
        String jsonInfo = jedis.get(phoneNum+mode);

        if (jsonInfo != null) {
            JSONObject jsonObject = JSON.parseObject(jsonInfo);
            code = jsonObject.getString("verificationCode");
        }

        jedis.close();
        return code;
    }

    //api 注册
    @Override
    @Transactional
    public ResponseBean signUp(String phoneNum, int mode, String code, String password, String name) {
        try{
            List<AccountSt> accountList = um.selectAccounts(phoneNum);
            if (accountList.size()>0){
                return ResponseBean.fail("当前账号已注册");
            }
            if (!checkCode(phoneNum, mode).equals(code)){
                return ResponseBean.fail("验证码错误");
            }else{
                //插入新用户
                AccountSt newAccountSt = new AccountSt();
                newAccountSt.setPhoneNum(phoneNum);
                newAccountSt.setPasswordDetail(password);
                newAccountSt.setNickName(name);
                um.insertNewUser(newAccountSt);
//                um.insertRelationship(newAccountSt.getAccountId(),4L);
                return ResponseBean.success();
            }
        }catch (Exception e){
            throw new MyException("注册失败，错误："+e);
        }
    }

    //api 重设密码
    @Override
    public ResponseBean resetPassword(String phoneNum, int mode, String code, String newPassword) {
        try{
            List<AccountSt> accountList = um.selectAccounts(phoneNum);
            if (accountList.size() < 1){
                return ResponseBean.fail("该账号还未注册");
            }
            if (!checkCode(phoneNum, mode).equals(code)){
                return ResponseBean.fail("验证码错误");
            }else{
                um.updatePassword(phoneNum, newPassword);
                return ResponseBean.success();
            }
        }catch (Exception e){
            throw new MyException("重设密码失败，错误："+e);
        }
    }
}
