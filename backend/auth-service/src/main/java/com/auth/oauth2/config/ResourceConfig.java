package com.auth.oauth2.config;

import com.alibaba.fastjson.JSONObject;
import com.example.common.enums.UserType;
import org.apache.catalina.User;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.security.config.annotation.method.configuration.EnableGlobalMethodSecurity;
import org.springframework.security.config.annotation.web.builders.HttpSecurity;
import org.springframework.security.core.Authentication;
import org.springframework.security.oauth2.config.annotation.web.configuration.EnableResourceServer;
import org.springframework.security.oauth2.config.annotation.web.configuration.ResourceServerConfigurerAdapter;
import org.springframework.security.oauth2.config.annotation.web.configurers.ResourceServerSecurityConfigurer;
import org.springframework.security.oauth2.provider.expression.OAuth2WebSecurityExpressionHandler;
import org.springframework.security.oauth2.provider.token.TokenStore;

@Configuration
@EnableResourceServer
@EnableGlobalMethodSecurity(prePostEnabled = true)
public class ResourceConfig extends ResourceServerConfigurerAdapter {

    @Override
    public void configure(HttpSecurity http) throws Exception {
        http
                .authorizeRequests()
                .antMatchers("/sign/**").permitAll() // 允许所有用户访问注册、登录、更改密码、验证码、获取token
                .antMatchers("/**").authenticated()
//                .antMatchers("/getUserId").hasAnyAuthority(
//                        UserType.Controller.getType()
//                )
//                .antMatchers("/controller/**").hasAnyAuthority(
//                        UserType.Controller.getType(),
//                        UserType.Manager.getType(),
//                        UserType.SuperManager.getType()
//                )
//                .antMatchers("/manager/**").hasAnyAuthority(
//                        UserType.Manager.getType(),
//                        UserType.SuperManager.getType()
//                )
//                .antMatchers("/superManager/**").hasAnyAuthority(
//                        UserType.SuperManager.getType()
//                )
                .anyRequest().authenticated()
                .and()
                .formLogin().disable()
                .csrf().disable(); // 禁用 CSRF，可能需要根据实际情况启用
    }

    @Override
    public void configure(ResourceServerSecurityConfigurer resources) throws Exception {
        resources.tokenStore(jwtTokenStore);
        resources.expressionHandler(new OAuth2WebSecurityExpressionHandler());
    }

    @Autowired
    TokenStore jwtTokenStore;
}
