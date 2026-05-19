OPENQASM 2.0;
include "qelib1.inc";
qreg q953[4];
cx q953[3],q953[2];
rz(3*pi/2) q953[3];
cx q953[2],q953[3];
cx q953[2],q953[1];
cx q953[0],q953[1];
