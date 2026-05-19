OPENQASM 2.0;
include "qelib1.inc";
qreg q825[6];
cx q825[4],q825[5];
cx q825[3],q825[4];
cx q825[2],q825[3];
cx q825[2],q825[1];
cx q825[0],q825[1];
