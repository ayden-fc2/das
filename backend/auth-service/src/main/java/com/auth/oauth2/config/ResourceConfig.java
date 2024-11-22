package com.auth.oauth2.config;

import com.example.common.enums.UserType;
import org.apache.catalina.User;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.security.config.annotation.web.builders.HttpSecurity;
import org.springframework.security.oauth2.config.annotation.web.configuration.EnableResourceServer;
import org.springframework.security.oauth2.config.annotation.web.configuration.ResourceServerConfigurerAdapter;
import org.springframework.security.oauth2.config.annotation.web.configurers.ResourceServerSecurityConfigurer;
import org.springframework.security.oauth2.provider.token.TokenStore;
import org.springframework.security.oauth2.provider.token.store.JwtAccessTokenConverter;
import org.springframework.security.oauth2.provider.token.store.JwtTokenStore;

@Configuration
@EnableResourceServer
public class ResourceConfig extends ResourceServerConfigurerAdapter {
    @Override
    public void configure(HttpSecurity http) throws Exception {
        http
                .authorizeRequests()
                .antMatchers("/sign/**").permitAll() // 允许所有用户访问注册、登录、更改密码、验证码、获取token
                .antMatchers("/test").hasAnyAuthority(
                        UserType.Observer.getType(),
                        UserType.Controller.getType(),
                        UserType.Manager.getType(),
                        UserType.SuperManager.getType()
                        )
                .antMatchers("/getUserId").hasAnyAuthority(
                        UserType.Observer.getType(),
                        UserType.Controller.getType(),
                        UserType.Manager.getType(),
                        UserType.SuperManager.getType()
                )
                .antMatchers("/controller/**").hasAnyAuthority(
                        UserType.Controller.getType(),
                        UserType.Manager.getType(),
                        UserType.SuperManager.getType()
                )
                .antMatchers("/manager/**").hasAnyAuthority(
                        UserType.Manager.getType(),
                        UserType.SuperManager.getType()
                )
                .antMatchers("/superManager/**").hasAnyAuthority(
                        UserType.SuperManager.getType()
                )
                .anyRequest().authenticated()
                .and()
                .formLogin().disable()
                .csrf().disable(); // 禁用 CSRF，可能需要根据实际情况启用
    }

    @Override
    public void configure(ResourceServerSecurityConfigurer resources) throws Exception {
        resources.tokenStore(jwtTokenStore);
    }

    @Autowired
    TokenStore jwtTokenStore;
}
