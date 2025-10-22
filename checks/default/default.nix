{ inputs, pkgs, ... }:
pkgs.testers.runNixOSTest rec {
  name = builtins.baseNameOf ./.;
  nodes.machine = {
    environment.systemPackages = [ inputs.self.packages.${pkgs.stdenv.system}.${name} ];
    virtualisation.memorySize = 2048;
  };
  testScript = ''machine.succeed("DEBUG=1 ${name}")'';
}
