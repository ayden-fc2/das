package com.zuul.gateway.config;

import org.springframework.context.annotation.Configuration;
import org.springframework.core.annotation.Order;
import org.springframework.security.config.annotation.web.builders.HttpSecurity;
import org.springframework.security.config.annotation.web.configuration.EnableWebSecurity;
import org.springframework.security.config.annotation.web.configuration.WebSecurityConfigurerAdapter;

@Configuration
@EnableWebSecurity
@Order(1)
public class ZuulSecurityConfig extends WebSecurityConfigurerAdapter {

    @Override
    protected void configure(HttpSecurity http) throws Exception {
        http
                .authorizeRequests()
                .antMatchers(
                        "/auth-service/**",
                        "/example2060/**",
                        "/dwg-handler/**",
                        "/file-manage/**"
                )
                .permitAll()
                .anyRequest()
                .authenticated()
                .and()
                .csrf().disable();  // 禁用 CSRF，可能需要根据实际情况启用

        // 添加下面这行配置，以确保OAuth2认证状态被正确传递
        // http.apply(new OAuth2Configurer());
    }

    /*private static class OAuth2Configurer extends SecurityConfigurerAdapter<DefaultSecurityFilterChain, HttpSecurity> {
        @Override
        public void configure(HttpSecurity http) {
            http
                    .addFilterBefore(new MyCustomFilter(), BasicAuthenticationFilter.class);
        }
    }

    private static class MyCustomFilter extends OncePerRequestFilter {
        @Override
        protected void doFilterInternal(HttpServletRequest request, HttpServletResponse response, FilterChain filterChain) throws ServletException, IOException {
            SecurityContext context = SecurityContextHolder.getContext();
            Authentication authentication = context.getAuthentication();

            // 在此处可以自定义逻辑处理认证信息，例如记录登录状态等

            filterChain.doFilter(request, response);
        }
    }*/
}

