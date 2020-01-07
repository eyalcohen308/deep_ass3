from xlsxwriter import worksheet
import pandas as pd

if __name__ == "__main__":
	a = [0.8229828850855746, 0.8148148148148148, 0.8107644840589647, 0.8143399409531843, 0.8976377952755905,
	     0.8995871180842279, 0.901344916882466, 0.9013880609724942, 0.9106012007566412, 0.9191077441077441,
	     0.9126777151510053, 0.9128145114101814, 0.9269851675817422, 0.9247258225324028, 0.9249809918053561,
	     0.9253364936933887, 0.9261705190077046, 0.9260829799617527, 0.9222867727540625, 0.9305897815015607]

	b = [0.7766870153440026, 0.7730837338730079, 0.7735272246550442, 0.7789072426937739, 0.8772162386081193,
	     0.8859110521075341, 0.8851526557925554, 0.8868100237047071, 0.9128919860627178, 0.9087462082912032,
	     0.9048143753178505, 0.9107568411537513, 0.9165383680414877, 0.9167364016736401, 0.9163934426229509,
	     0.9118087941372418, 0.9196180838909743, 0.9240904947377144, 0.9216077784529907, 0.9188314382936343]
	c = [0.9438502673796791, 0.9466622494616531, 0.9417702511053642, 0.9426258053282257, 0.9497370462970945,
	     0.9522178734507502, 0.9471488178025035, 0.9496738117427772, 0.9516264252179745, 0.9484578591456375,
	     0.9521717086361332, 0.9514207149404217, 0.9503534440243301, 0.9500913469523335, 0.9497325295066655,
	     0.9510248655913979, 0.9550202156334232, 0.9480268681780016, 0.9524975514201763, 0.9479544495993252]

	d = [0.9321465634230063, 0.9299565362754931, 0.9302228178924443, 0.9325479930191972, 0.9456366237482118,
	     0.9376897133220911, 0.9442254466157847, 0.9376585126401047, 0.9448304045770332, 0.9477929060792646,
	     0.9479132435096944, 0.9440214208016066, 0.9484084332368747, 0.9411093915451657, 0.9491611162633842,
	     0.9485243651338366, 0.9458425437850706, 0.9489888456565521, 0.9501456456468861, 0.9511185330713818]

	df = pd.DataFrame()
	df['Examples'] = [500/100 * i for i in range(1, len(a) + 1)]
	df['A'] = a
	df['B'] = b
	df['C'] = c
	df['D'] = d
	print(len(a), len(b))
	# Converting to excel
	df.to_excel('result.xlsx', index=False)
# worksheet.write_column('A', list)
