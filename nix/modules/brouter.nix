{ self }:
{ config, lib, pkgs, ... }:
let
  cfg = config.services.brouter;
  format = pkgs.formats.toml { };
  configFile = format.generate "brouter.toml" cfg.settings;
  defaultPackage = self.packages.${pkgs.stdenv.hostPlatform.system}.default;
in
{
  options.services.brouter = {
    enable = lib.mkEnableOption "brouter local LLM router";

    package = lib.mkOption {
      type = lib.types.package;
      default = defaultPackage;
      defaultText = lib.literalExpression "inputs.brouter.packages.${pkgs.stdenv.hostPlatform.system}.default";
      description = "brouter package to run.";
    };

    settings = lib.mkOption {
      type = format.type;
      default = { };
      example = lib.literalExpression ''
        {
          server = {
            host = "127.0.0.1";
            port = 8080;
          };
          router = {
            default_objective = "balanced";
            rules = [
              {
                name = "private-local";
                when_contains = [ "secret" "credentials" ];
                objective = "local_only";
                require_capabilities = [ "local" ];
              }
            ];
          };
          telemetry.database_path = "/var/lib/brouter/brouter.db";
          providers.ollama = {
            kind = "open-ai-compatible";
            base_url = "http://localhost:11434/v1";
          };
          models.fast_local = {
            provider = "ollama";
            model = "qwen2.5-coder:7b";
            context_window = 32768;
            capabilities = [ "chat" "code" "local" ];
          };
        }
      '';
      description = "Declarative brouter TOML settings.";
    };

    environmentFile = lib.mkOption {
      type = lib.types.nullOr lib.types.path;
      default = null;
      description = "Optional environment file containing provider API keys.";
    };

    openFirewall = lib.mkOption {
      type = lib.types.bool;
      default = false;
      description = "Open the configured TCP port in the firewall.";
    };
  };

  config = lib.mkIf cfg.enable {
    systemd.services.brouter = {
      description = "brouter local LLM router";
      wantedBy = [ "multi-user.target" ];
      after = [ "network-online.target" ];
      wants = [ "network-online.target" ];
      serviceConfig = {
        ExecStart = "${cfg.package}/bin/brouter serve --config ${configFile}";
        Restart = "on-failure";
        DynamicUser = true;
        StateDirectory = "brouter";
        RuntimeDirectory = "brouter";
      } // lib.optionalAttrs (cfg.environmentFile != null) {
        EnvironmentFile = cfg.environmentFile;
      };
    };

    networking.firewall.allowedTCPPorts = lib.mkIf cfg.openFirewall [ (cfg.settings.server.port or 8080) ];
  };
}
