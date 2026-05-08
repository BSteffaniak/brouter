{
  description = "brouter local LLM router";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    rust-overlay = {
      url = "github:oxalica/rust-overlay";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    cargoMacheteSrc = {
      url = "github:BSteffaniak/cargo-machete/ignored-dirs";
      flake = false;
    };
  };

  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
      rust-overlay,
      cargoMacheteSrc,
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = import nixpkgs {
          inherit system;
          overlays = [ (import rust-overlay) ];
        };
        rustToolchain = pkgs.rust-bin.stable.latest.default.override {
          extensions = [
            "rustfmt"
            "clippy"
            "rust-src"
          ];
        };
        rustPlatform = pkgs.makeRustPlatform {
          cargo = rustToolchain;
          rustc = rustToolchain;
        };
        cargoMachete = pkgs.rustPlatform.buildRustPackage {
          pname = "cargo-machete";
          version = "ignored-dirs";
          src = cargoMacheteSrc;
          cargoLock = {
            lockFile = "${cargoMacheteSrc}/Cargo.lock";
          };
          doCheck = false;
        };
        brouterPackage = rustPlatform.buildRustPackage {
          pname = "brouter";
          version = "0.1.0";
          src = ./.;
          cargoLock = {
            lockFile = ./Cargo.lock;
            allowBuiltinFetchGit = true;
          };
          nativeBuildInputs = with pkgs; [ pkg-config ];
          buildInputs =
            with pkgs;
            [ openssl ]
            ++ pkgs.lib.optionals pkgs.stdenv.isDarwin [ libiconv ];
          cargoBuildFlags = [
            "-p"
            "brouter_cli"
          ];
          cargoTestFlags = [
            "--workspace"
          ];
        };
      in
      {
        packages = {
          default = brouterPackage;
          brouter = brouterPackage;
        } // pkgs.lib.optionalAttrs pkgs.stdenv.isLinux {
          container = pkgs.dockerTools.buildLayeredImage {
            name = "brouter";
            tag = "latest";
            contents = [ brouterPackage ];
            config = {
              Entrypoint = [ "${brouterPackage}/bin/brouter" ];
              Cmd = [
                "serve"
                "--config"
                "/config/brouter.toml"
              ];
              ExposedPorts."8080/tcp" = { };
            };
          };
        };

        apps =
          let
            app = {
              type = "app";
              program = "${brouterPackage}/bin/brouter";
              meta.description = "Run the brouter CLI";
            };
          in
          {
            default = app;
            brouter = app;
          };

        checks = {
          package = brouterPackage;
          example-config = pkgs.runCommand "brouter-example-config-check" { } ''
            OPENAI_API_KEY=dummy ${brouterPackage}/bin/brouter check-config --strict --config ${./brouter.example.toml}
            touch $out
          '';
        };

        devShells.default = pkgs.mkShell {
          buildInputs =
            with pkgs;
            [
              rustToolchain
              cargo-deny
              cargoMachete
              pkg-config
              openssl
              fish
              git
            ]
            ++ pkgs.lib.optionals pkgs.stdenv.isDarwin [
              libiconv
            ];

          shellHook = ''
            echo "brouter development environment loaded"
            echo "Available tools:"
            echo "  - cargo ($(cargo --version))"
            echo "  - rustc ($(rustc --version))"
            echo "  - clippy ($(cargo clippy --version))"
            echo "  - cargo-deny ($(cargo deny --version))"
            echo "  - cargo-machete ($(cargo machete --version))"

            # Only exec fish if we're in an interactive shell (not running a command)
            if [ -z "$IN_NIX_SHELL_FISH" ] && [ -z "$BASH_EXECUTION_STRING" ]; then
              case "$-" in
                *i*) export IN_NIX_SHELL_FISH=1; exec fish ;;
              esac
            fi
          '';
        };
      }
    )
    // {
      nixosModules.default = import ./nix/modules/brouter.nix { inherit self; };
      homeManagerModules.default = import ./nix/home-manager/brouter.nix { inherit self; };
    };
}
