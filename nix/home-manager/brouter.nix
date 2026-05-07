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
      description = "Declarative brouter TOML settings.";
    };

    environmentFile = lib.mkOption {
      type = lib.types.nullOr lib.types.path;
      default = null;
      description = "Optional environment file containing provider API keys.";
    };
  };

  config = lib.mkIf cfg.enable {
    systemd.user.services.brouter = {
      Unit = {
        Description = "brouter local LLM router";
        After = [ "network-online.target" ];
      };
      Service = {
        ExecStart = "${cfg.package}/bin/brouter serve --config ${configFile}";
        Restart = "on-failure";
      } // lib.optionalAttrs (cfg.environmentFile != null) {
        EnvironmentFile = cfg.environmentFile;
      };
      Install.WantedBy = [ "default.target" ];
    };
  };
}
