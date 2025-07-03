import smtplib
import os
import logging
import threading
import time
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

@dataclass
class EmailConfig:
    smtp_server: str = "smtp.gmail.com"
    smtp_port: int = 587  # Mudará para 465 se necessário
    email_user: str = ""
    email_password: str = ""
    enabled: bool = True
    retry_attempts: int = 3
    retry_delay: float = 2.0
    admin_email: str = "dev.acnormun@gmail.com"
    use_ssl: bool = False  # Nova opção para SSL direto

@dataclass
class ProcessoData:
    numero_processo: str
    tema: str
    data_distribuicao: str
    responsavel: str
    status: str
    comentarios: str = ""
    suspeitos: List[str] = None
    ultima_atualizacao: str = ""

    def __post_init__(self):
        if self.suspeitos is None:
            self.suspeitos = []
        if not self.ultima_atualizacao:
            self.ultima_atualizacao = datetime.now().strftime('%d/%m/%Y às %H:%M:%S')

class EmailNotificationService:
    
    def __init__(self):
        self.config = self._load_config()
        self.email_mappings = self._load_email_mappings()
        self._validate_config()
        self._email_queue = []
        self._is_processing = False
        logger.info(f"EmailNotificationService iniciado - {'Habilitado' if self.config.enabled else 'Desabilitado'}")
        
        # Log da configuração (sem senha)
        if self.config.enabled:
            logger.info(f"Configuração: {self.config.email_user} -> SMTP {self.config.smtp_server}:{self.config.smtp_port}")

    def _load_config(self) -> EmailConfig:
        # Verifica se deve usar SSL (porta 465)
        use_ssl = os.getenv('EMAIL_USE_SSL', 'false').lower() == 'true'
        
        return EmailConfig(
            email_user=os.getenv('EMAIL_USER', ''),
            email_password=os.getenv('EMAIL_APP_PASSWORD', ''),
            enabled=os.getenv('EMAIL_NOTIFICATIONS', 'true').lower() == 'true',
            retry_attempts=int(os.getenv('EMAIL_RETRY_ATTEMPTS', '3')),
            retry_delay=float(os.getenv('EMAIL_RETRY_DELAY', '2')),
            admin_email=os.getenv('ADMIN_EMAIL', 'dev.acnormun@gmail.com'),
            smtp_port=465 if use_ssl else 587,
            use_ssl=use_ssl
        )

    def _load_email_mappings(self) -> Dict[str, str]:
        mappings = {
            'NATÁLIA': os.getenv('EMAIL_NATALIA', 'natalia@escritorio.com'),
            'NAIRA': os.getenv('EMAIL_NAIRA', 'naira@escritorio.com'),
            'JOÃO': os.getenv('EMAIL_JOAO', 'joao@escritorio.com'),
            'MARIA': os.getenv('EMAIL_MARIA', 'maria@escritorio.com'),
            'PEDRO': os.getenv('EMAIL_PEDRO', 'pedro@escritorio.com'),
            'ANA': os.getenv('EMAIL_ANA', 'ana@escritorio.com'),
            'LUCAS': os.getenv('EMAIL_LUCAS', 'lucas@escritorio.com'),
            'CARLA': os.getenv('EMAIL_CARLA', 'carla@escritorio.com'),
        }
        return {k: v for k, v in mappings.items() if v and '@' in v}

    def _validate_config(self) -> None:
        if not self.config.enabled:
            logger.info("Notificações por email desabilitadas")
            return
        if not self.config.email_user or not self.config.email_password:
            logger.warning("Credenciais de email não configuradas. Notificações desabilitadas.")
            logger.warning("Configure EMAIL_USER e EMAIL_APP_PASSWORD no .env")
            self.config.enabled = False
            return
        if '@' not in self.config.email_user:
            logger.warning("EMAIL_USER inválido. Notificações desabilitadas.")
            self.config.enabled = False
            return
        logger.info(f"Configuração de email válida para {self.config.email_user}")

    def get_email_por_responsavel(self, responsavel: str) -> str:
        if not responsavel:
            return self.config.admin_email
        email = self.email_mappings.get(responsavel.upper().strip())
        if not email:
            logger.warning(f"Email não encontrado para responsável: '{responsavel}'. Usando admin.")
            return self.config.admin_email
        return email

    def get_prioridade_info(self, tema: str) -> Tuple[str, str, str]:
        tema_upper = tema.upper()
        alta_prioridade = ['URGENTE', 'EMERGENCIAL', 'LIMINAR', 'MANDADO', 'HABEAS']
        normal_prioridade = ['SAÚDE', 'PREVIDENCIÁRIO', 'TRABALHISTA', 'CRIMINAL']
        if any(palavra in tema_upper for palavra in alta_prioridade):
            return 'ALTA', '#dc3545', ''
        elif any(palavra in tema_upper for palavra in normal_prioridade):
            return 'NORMAL', '#ffc107', ''
        else:
            return 'BAIXA', '#28a745', ''

    def create_html_email(self, processo: ProcessoData) -> str:
        prioridade, cor_prioridade, icone = self.get_prioridade_info(processo.tema)
        suspeitos_formatados = ', '.join(processo.suspeitos) if processo.suspeitos else 'Nenhum suspeito detectado'
        comentarios_formatados = processo.comentarios if processo.comentarios.strip() else 'Sem comentários adicionais'
        html = f"""
<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Novo Processo Atribuído</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            margin: 0;
            padding: 0;
            background-color: #f4f4f4;
        }}
        .container {{
            max-width: 650px;
            margin: 20px auto;
            background: white;
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }}
        .header {{
            background: linear-gradient(135deg, #2c5aa0 0%, #1e3f73 100%);
            color: white;
            padding: 25px;
            text-align: center;
        }}
        .header h1 {{
            margin: 0;
            font-size: 26px;
            font-weight: 600;
        }}
        .priority-badge {{
            background: {cor_prioridade};
            color: white;
            padding: 8px 16px;
            border-radius: 20px;
            font-size: 14px;
            font-weight: bold;
            margin-top: 10px;
            display: inline-block;
        }}
        .content {{
            padding: 30px;
            border-left: 6px solid {cor_prioridade};
        }}
        .field {{
            margin: 18px 0;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 8px;
            border-left: 4px solid #e9ecef;
        }}
        .field.highlight {{
            background: #e3f2fd;
            border-left-color: #2196f3;
        }}
        .field.warning {{
            background: #fff3e0;
            border-left-color: #ff9800;
        }}
        .label {{
            font-weight: 600;
            color: #2c5aa0;
            display: block;
            margin-bottom: 5px;
            font-size: 14px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        .value {{
            font-size: 16px;
            color: #333;
            word-wrap: break-word;
        }}
        .value.large {{
            font-size: 18px;
            font-weight: 600;
            color: #1e3f73;
        }}
        .comentarios {{
            background: #f0f0f0;
            padding: 15px;
            border-radius: 6px;
            font-style: italic;
            margin-top: 8px;
            border-left: 3px solid #ccc;
        }}
        .suspeitos {{
            background: #ffebee;
            padding: 15px;
            border-radius: 6px;
            margin-top: 8px;
            border-left: 3px solid #f44336;
        }}
        .suspeitos.empty {{
            background: #e8f5e8;
            border-left-color: #4caf50;
        }}
        .footer {{
            background: #f8f9fa;
            padding: 25px;
            text-align: center;
            color: #666;
            font-size: 13px;
            border-top: 1px solid #eee;
        }}
        .footer strong {{
            color: #2c5aa0;
            font-size: 16px;
        }}
        .divider {{
            height: 2px;
            background: linear-gradient(90deg, {cor_prioridade}, transparent);
            margin: 20px 0;
        }}
        @media (max-width: 600px) {{
            .container {{
                margin: 10px;
                border-radius: 5px;
            }}
            .content {{
                padding: 20px;
            }}
            .header h1 {{
                font-size: 22px;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Novo Processo Atribuído</h1>
            <span class="priority-badge">PRIORIDADE: {prioridade}</span>
        </div>
        
        <div class="content">
            <div class="field highlight">
                <span class="label">Número do Processo</span>
                <span class="value large">{processo.numero_processo}</span>
            </div>
            
            <div class="field">
                <span class="label">Tema</span>
                <span class="value">{processo.tema}</span>
            </div>
            
            <div class="field">
                <span class="label">Data de Distribuição</span>
                <span class="value">{processo.data_distribuicao}</span>
            </div>
            
            <div class="field highlight">
                <span class="label">Responsável</span>
                <span class="value large">{processo.responsavel}</span>
            </div>
            
            <div class="field">
                <span class="label">Status</span>
                <span class="value">{processo.status}</span>
            </div>
            
            <div class="divider"></div>
            
            <div class="field">
                <span class="label">Comentários</span>
                <div class="comentarios">{comentarios_formatados}</div>
            </div>
            
            <div class="field {'warning' if processo.suspeitos else ''}">
                <span class="label">Análise de Suspeição</span>
                <div class="suspeitos {'empty' if not processo.suspeitos else ''}">
                    <strong>{'Suspeitos Detectados:' if processo.suspeitos else 'Nenhum impedimento detectado'}</strong><br>
                    {suspeitos_formatados}
                </div>
            </div>
            
            <div class="field">
                <span class="label">Data/Hora de Atribuição</span>
                <span class="value">{processo.ultima_atualizacao}</span>
            </div>
        </div>
        
        <div class="footer">
            <p><strong>Sistema de Gestão de Processos Jurídicos</strong></p>
            <p>Esta é uma notificação automática do sistema de triagem.</p>
            <p>Por favor, não responda este email. Em caso de dúvidas, contate o administrador.</p>
            <p style="margin-top: 15px; color: #999; font-size: 11px;">
                Email enviado em {datetime.now().strftime('%d/%m/%Y às %H:%M:%S')}
            </p>
        </div>
    </div>
</body>
</html>
        """
        return html

    def send_email_smtp(self, destinatario: str, assunto: str, html_content: str) -> bool:
        """Envia email com fallback automático entre STARTTLS e SSL"""
        
        # Primeira tentativa: STARTTLS (porta 587)
        if not self.config.use_ssl:
            try:
                logger.info(f"Tentando STARTTLS (porta 587) para {destinatario}")
                msg = MIMEMultipart('alternative')
                msg['From'] = self.config.email_user
                msg['To'] = destinatario
                msg['Subject'] = assunto
                html_part = MIMEText(html_content, 'html', 'utf-8')
                msg.attach(html_part)
                
                with smtplib.SMTP(self.config.smtp_server, 587) as server:
                    server.set_debuglevel(0)
                    server.starttls()
                    server.login(self.config.email_user, self.config.email_password)
                    server.send_message(msg)
                    
                logger.info("Email enviado com sucesso via STARTTLS")
                return True
                
            except Exception as e:
                logger.warning(f"STARTTLS falhou: {e}")
                logger.info("Tentando fallback para SSL...")
        
        # Segunda tentativa: SSL direto (porta 465)
        try:
            logger.info(f"Tentando SSL (porta 465) para {destinatario}")
            msg = MIMEMultipart('alternative')
            msg['From'] = self.config.email_user
            msg['To'] = destinatario
            msg['Subject'] = assunto
            html_part = MIMEText(html_content, 'html', 'utf-8')
            msg.attach(html_part)
            
            with smtplib.SMTP_SSL(self.config.smtp_server, 465) as server:
                server.set_debuglevel(0)
                server.login(self.config.email_user, self.config.email_password)
                server.send_message(msg)
                
            logger.info("Email enviado com sucesso via SSL")
            
            # Se SSL funcionou, salva a configuração para próximas vezes
            if not self.config.use_ssl:
                logger.info("Salvando preferência por SSL para próximas tentativas")
                self.config.use_ssl = True
                self.config.smtp_port = 465
                
            return True
            
        except smtplib.SMTPAuthenticationError as e:
            logger.error(f"Erro de autenticação SMTP: {e}")
            logger.error("Verifique se EMAIL_APP_PASSWORD está configurado corretamente")
            return False
        except smtplib.SMTPException as e:
            logger.error(f"Erro SMTP: {e}")
            return False
        except Exception as e:
            logger.error(f"Erro inesperado no envio: {e}")
            return False

    def send_notification_with_retry(self, processo: ProcessoData) -> bool:
        if not self.config.enabled:
            logger.debug("Notificações desabilitadas")
            return True
        destinatario = self.get_email_por_responsavel(processo.responsavel)
        prioridade, _, icone = self.get_prioridade_info(processo.tema)
        prefixo = f'URGENTE - ' if prioridade == 'ALTA' else ''
        assunto = f"{prefixo}Novo Processo: {processo.numero_processo}"
        html_content = self.create_html_email(processo)
        
        for tentativa in range(1, self.config.retry_attempts + 1):
            logger.info(f"Tentativa {tentativa}/{self.config.retry_attempts} - Enviando para {processo.responsavel} ({destinatario})")
            success = self.send_email_smtp(destinatario, assunto, html_content)
            if success:
                logger.info(f"Email enviado com sucesso na tentativa {tentativa}")
                return True
            if tentativa < self.config.retry_attempts:
                logger.warning(f"Tentativa {tentativa} falhou, aguardando {self.config.retry_delay}s...")
                time.sleep(self.config.retry_delay)
                
        logger.error(f"Falha em todas as {self.config.retry_attempts} tentativas para {processo.responsavel}")
        self._send_error_notification(processo, destinatario)
        return False

    def _send_error_notification(self, processo: ProcessoData, destinatario_original: str) -> None:
        try:
            erro_processo = ProcessoData(
                numero_processo=f"ERRO - {processo.numero_processo}",
                tema="FALHA NA NOTIFICAÇÃO POR EMAIL",
                data_distribuicao=processo.data_distribuicao,
                responsavel=f"ERRO PARA: {processo.responsavel}",
                status="FALHA NO ENVIO",
                comentarios=f"Falha ao enviar notificação para {destinatario_original}.\n\nProcesso original:\n- Número: {processo.numero_processo}\n- Tema: {processo.tema}\n- Responsável: {processo.responsavel}\n\nVerifique configurações de email e conectividade.",
                suspeitos=["Erro no sistema de notificação"]
            )
            html_erro = self.create_html_email(erro_processo)
            self.send_email_smtp(
                self.config.admin_email,
                f"ERRO: Falha na notificação - {processo.numero_processo}",
                html_erro
            )
        except Exception as e:
            logger.error(f"Erro ao enviar notificação de falha para admin: {e}")

    def send_notification_async(self, processo: ProcessoData) -> None:
        def _send():
            try:
                self.send_notification_with_retry(processo)
            except Exception as e:
                logger.error(f"Erro na thread de notificação: {e}")
        thread = threading.Thread(target=_send, daemon=True)
        thread.start()

    def test_configuration(self) -> Dict:
        result = {
            'enabled': self.config.enabled,
            'email_configured': bool(self.config.email_user and self.config.email_password),
            'responsaveis_mapeados': len(self.email_mappings),
            'emails_mapeados': list(self.email_mappings.keys()),
            'admin_email': self.config.admin_email,
            'connectivity': False,
            'smtp_server': self.config.smtp_server,
            'smtp_port': self.config.smtp_port,
            'use_ssl': self.config.use_ssl,
            'message': ''
        }
        if not self.config.enabled:
            result['message'] = 'Notificações por email estão desabilitadas'
            return result
        if not result['email_configured']:
            result['message'] = 'Credenciais SMTP não configuradas (EMAIL_USER e EMAIL_APP_PASSWORD)'
            return result
            
        # Testa conectividade com fallback
        try:
            # Tenta STARTTLS primeiro
            with smtplib.SMTP(self.config.smtp_server, 587) as server:
                server.starttls()
                server.login(self.config.email_user, self.config.email_password)
                result['connectivity'] = True
                result['message'] = 'Configuração válida e conectividade OK (STARTTLS)'
                result['smtp_port'] = 587
                return result
        except Exception:
            pass
            
        try:
            # Tenta SSL se STARTTLS falhou
            with smtplib.SMTP_SSL(self.config.smtp_server, 465) as server:
                server.login(self.config.email_user, self.config.email_password)
                result['connectivity'] = True
                result['message'] = 'Configuração válida e conectividade OK (SSL)'
                result['smtp_port'] = 465
                result['use_ssl'] = True
                return result
        except Exception as e:
            result['message'] = f'Erro de conectividade SMTP: {str(e)}'
            
        return result

    def send_test_email(self, destinatario: Optional[str] = None) -> bool:
        if not destinatario:
            destinatario = self.config.admin_email
        processo_teste = ProcessoData(
            numero_processo=f'TESTE-{int(time.time())}',
            tema='TESTE AUTOMATIZADO DO SISTEMA',
            data_distribuicao=datetime.now().strftime('%d/%m/%Y'),
            responsavel='SISTEMA',
            status='TESTE',
            comentarios='Este é um email de teste do sistema de notificações. Se você recebeu este email, a configuração está funcionando corretamente.',
            suspeitos=['João da Silva Teste', 'Maria Santos Exemplo']
        )
        return self.send_notification_with_retry(processo_teste)

email_service = EmailNotificationService()

def enviar_notificacao_processo(dados_processo: Dict) -> bool:
    try:
        processo = ProcessoData(
            numero_processo=dados_processo.get('numero', ''),
            tema=dados_processo.get('tema', ''),
            data_distribuicao=dados_processo.get('data_dist', ''),
            responsavel=dados_processo.get('responsavel', ''),
            status=dados_processo.get('status', ''),
            comentarios=dados_processo.get('comentarios', ''),
            suspeitos=dados_processo.get('suspeitos', [])
        )
        email_service.send_notification_async(processo)
        return True
    except Exception as e:
        logger.error(f"Erro ao preparar notificação: {e}")
        return False